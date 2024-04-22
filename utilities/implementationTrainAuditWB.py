# Imports and configuration
import argparse
import os
import copy
import torch.utils
import torch.utils.data
from fastDP import PrivacyEngine
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import timm
from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")

# Main function for training and auditing
def main(args):

    # Set torch device
    device=torch.device("cuda:0")

    # Load auditing parameters
    m = args.m
    k_plus = args.k_plus
    k_minus = args.k_minus

    # Data transformations and loading
    data_transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'none': torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

######################### TODO implementation start

    # Load test and train datasets
    data_dir = '/s/lovelace/c/nobackup/iray/dp-imgclass/PediatricChestX-rayPneumoniaData'
    trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    trainset_noaug = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['none'])
    testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])

    # TODO Add m canaries to the beginning of trainset and trainset_noaug
    here ...
    all_indices = torch.randperm(len(trainset))
    canary_indices = all_indices[:m]
    non_canary_indices = all_indices[m:]

    # Initialize Si for canaries to -1 or 1 with equal probability
    Si = torch.randint(0, 2, (len(trainset),)) * 2 - 1
    Si[non_canary_indices] = 1

    # Find which indices will be in x_IN and x_OUT
    x_IN_indices = torch.where(Si == 1)[0].tolist()
    x_OUT_indices = torch.where(Si == -1)[0].tolist()
    x_IN_set = torch.utils.data.Subset(trainset, x_IN_indices)
    x_OUT_set = torch.utils.data.Subset(trainset, x_OUT_indices)

    # Set up train loader (also thought of as x_IN loader) and the test loader, no x_OUT loader needed
    trainloader = torch.utils.data.DataLoader(x_IN_set, batch_size=args.mini_bs, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    # Set up canary loader for later auditing
    # TODO email authors about augmentations for canaries, just running no augmentations for now
    canary_set = torch.utils.data.Subset(trainset_noaug, canary_indices)
    canary_loader = torch.utils.data.DataLoader(canary_set, batch_size=1, shuffle=False, num_workers=4)

######################### TODO implementation end

    # Handle for gradient accumulation steps
    n_acc_steps = args.bs // args.mini_bs

    # Create model and validate
    net = timm.create_model(args.model, pretrained = True, num_classes = 2)
    net = ModuleValidator.fix(net).to(device)

    # Create optimizer and loss functions
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # Modify layers for BiTFiT
    if 'BiTFiT' in args.clipping_mode:
        for name,layer in net.named_modules():
            if hasattr(layer,'weight'):
                temp_layer=layer
        for name,param in net.named_parameters():
            if '.bias' not in name:
                param.requires_grad_(False)
        for param in temp_layer.parameters():
            param.requires_grad_(True)

    # Set up privacy engine, but use x_IN_set instead of full_trainset since not all training samples are used
    if 'nonDP' not in args.clipping_mode:
        # Calculate and display the noise level
        sigma = get_noise_multiplier(target_epsilon = args.epsilon,
                                     target_delta = 1e-5, # TODO make this an argument?
                                     sample_rate = args.bs / len(x_IN_set),
                                     epochs = args.epochs)
        print(f'adding noise level {sigma}')

        # Build privacy engine and attach it to the current optimizer
        privacy_engine = PrivacyEngine(
            net,
            batch_size=args.bs,
            sample_size=len(x_IN_set),
            noise_multiplier=sigma,
            epochs=args.epochs,
            clipping_mode=args.clipping_mode,
            clipping_style=args.clipping_style
        )
        privacy_engine.attach(optimizer)

    # Lists to store metrics
    tr_loss = []
    te_loss = []
    tr_acc = []
    te_acc = []

    def train(epoch):
        # Run training logic on train samples
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                optimizer.step()
                optimizer.zero_grad()
            train_loss += loss.item()
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

        # Store and print loss and accuracy information
        tr_loss.append(train_loss / (batch_idx + 1))
        tr_acc.append(100. * correct / total)
        print('Epoch: ', epoch, 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def test(epoch):
        # Run testing logic on test samples
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                total += targets.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

            # Store and print loss and accuracy information
            te_loss.append(test_loss / (batch_idx + 1))
            te_acc.append(100. * correct / total)
            print('Epoch: ', epoch, 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

######################### TODO implementation start

    # Run train and test functions for number of epochs, but save initial state of model first
    model_states = []
    model_states.append(copy.deepcopy(net.state_dict()))
    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)
        # Save the state of the current model
        model_states.append(copy.deepcopy(net.state_dict()))

    # Save final state of model after training
    model_states.append(copy.deepcopy(net.state_dict()))








    optimizer.


# we need to steal per sample gradients and their norms from the priv engine before noise is added

    privacy_engine

    # Compute scores from each of the model weights for each canary sample
    scores = []
    net.eval()
    clipping_fn = privacy_engine
    for i in range(1, len(model_states)):
        # Load the model state at epoch i-1
        net.load_state_dict(model_states[i - 1])
        initial_gradients = []

        # Compute gradients for the initial state
        for inputs, targets in canary_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True
            
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Apply clipping using the clipping function from PrivacyEngine
            clipping_fn(model)

            # Save the clipped gradients
            initial_gradients.append(inputs.grad.clone())

        # Now compute the influence score using the state at time i
        model.load_state_dict(model_states[i])

        # Compare against gradients from the previous state
        for j, (inputs, targets) in enumerate(canary_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True
            
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Apply clipping to these new gradients
            clipping_fn(model)

            # Calculate the score based on the dot product of the clipped gradient difference and weight update
            delta_weights = {name: model_states[i][name] - model_states[i-1][name] for name in model_states[i]}
            score = sum(
                (inputs.grad.flatten() * delta_weights[name].flatten()).sum()
                for name in delta_weights
            )
            scores.append(score.item())
    Y = torch.tensor(scores)
















    # Sort the scores to make guesses
    sorted_indices = torch.argsort(Y, descending=True)
    T = torch.zeros(m)
    T[sorted_indices[:k_plus]] = 1
    T[sorted_indices[-k_minus:]] = -1

    # Load the true selection for the canaries
    S = Si[canary_indices]

    # Compare S and T to see how many guesses were correct, do not guess for zero values
    guess_indices = torch.where(T != 0)[0]
    correct_guesses = 0
    for idx in guess_indices:
        if T[idx] == S[idx]:
            correct_guesses += 1
    print("Correct Guesses: " + str(correct_guesses))

######################### TODO implementation end

    # Print final metrics
    print(tr_loss, tr_acc, te_loss, te_acc)

if __name__ == '__main__':
    # Create and parse arguments
    parser = argparse.ArgumentParser(description='Image Classification and Privacy Auditing')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--epochs', default=5, type=int, help='numter of epochs')
    parser.add_argument('--bs', default=250, type=int, help='batch size')
    parser.add_argument('--mini_bs', type=int, default=250)
    parser.add_argument('--epsilon', default=1, type=float, help='target epsilon')
    parser.add_argument('--clipping_mode', type=str, default='MixOpt', choices=['BiTFiT', 'MixOpt', 'nonDP', 'nonDP-BiTFiT'])
    parser.add_argument('--clipping_style', default='all-layer', nargs='+', type=str)
    parser.add_argument('--model', default='beit_base_patch16_224.in22k_ft_in22k', type=str, help='model name')
    parser.add_argument('--m', type=int, default=500, help='number of auditing examples')
    parser.add_argument('--k_plus', type=int, default=25, help='number of positive guesses')
    parser.add_argument('--k_minus', type=int, default=25, help='number of negative guesses')
    args = parser.parse_args()

    # Run main function
    main(args)