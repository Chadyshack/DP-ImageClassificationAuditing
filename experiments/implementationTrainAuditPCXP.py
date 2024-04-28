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
import random
import warnings; warnings.filterwarnings("ignore")

# Main function for training and auditing
def main(args):

    # Set torch device
    device=torch.device("cuda:0")

    # Load auditing parameters
    m = args.m
    k_plus = args.k_plus
    k_minus = args.k_minus
    flip_canaries = args.flip_canaries
    n = args.n

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
        'canary': torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load test and train datasets (denoting as full because this is reduced later)
    data_dir = '/s/lovelace/c/nobackup/iray/dp-imgclass/PediatricChestX-rayPneumoniaData'
    full_trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
    full_trainset_canaryaug = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['canary'])

    # Specify canary and non-canary indices within train dataset randomly
    all_indices = torch.randperm(len(full_trainset))
    canary_indices = all_indices[:m]
    non_canary_indices = all_indices[n:]

    # Flip the labels for the canaries if flipping
    if flip_canaries:
        # Find all possible labels
        all_labels = list(range(2))
        for index in canary_indices:
            # Find the current label, make list to randomly select from not including it
            current_label = full_trainset.samples[index][1]
            new_labels = [label for label in all_labels if label != current_label]

            # Randomly select a new label
            new_label = random.choice(new_labels)

            # Update both full trainsets with this flipped label
            full_trainset.samples[index] = (full_trainset.samples[index][0], new_label)
            full_trainset_canaryaug.samples[index] = (full_trainset_canaryaug.samples[index][0], new_label)    

    # Initialize Si for canaries to -1 or 1 with equal probability
    Si = torch.randint(0, 2, (len(full_trainset),)) * 2 - 1
    Si[non_canary_indices] = 1

    # Find which indices will be in x_IN and x_OUT
    x_IN_indices = torch.where(Si == 1)[0].tolist()
    x_OUT_indices = torch.where(Si == -1)[0].tolist()
    x_IN_set = torch.utils.data.Subset(full_trainset, x_IN_indices)
    x_OUT_set = torch.utils.data.Subset(full_trainset, x_OUT_indices)

    # Set up train loader (also thought of as x_IN loader) and the test loader, no x_OUT loader needed
    trainloader = torch.utils.data.DataLoader(x_IN_set, batch_size=args.mini_bs, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    # Set up canary loader for later auditing
    canary_set = torch.utils.data.Subset(full_trainset_canaryaug, canary_indices)
    canary_loader = torch.utils.data.DataLoader(canary_set, batch_size=100, shuffle=False, num_workers=4)

    # Handle for gradient accumulation steps
    n_acc_steps = args.bs // args.mini_bs

    # Create model and validate
    net = timm.create_model(args.model, pretrained = True, num_classes = 2)
    net = ModuleValidator.fix(net).to(device)

    # Create optimizer and loss functions
    criterion = nn.CrossEntropyLoss()
    canary_criterion = nn.CrossEntropyLoss(reduction='none')
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
                                     target_delta = 1e-5,
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

    def compute_loss_for_canaries():
        # Function to compute loss for each canary
        net.eval()
        losses = []
        with torch.no_grad():
            for inputs, targets in canary_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = canary_criterion(outputs, targets)
                losses.extend(loss.tolist())
        return losses

    # Run train and test functions for number of epochs, but save initial state of model first
    w0 = copy.deepcopy(net.state_dict())
    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)

    # Save final state of model after training
    wL = copy.deepcopy(net.state_dict())

    # Compute loss for canaries with inital and final weights, use information to compute scores
    net.load_state_dict(w0)
    initial_losses = compute_loss_for_canaries()
    net.load_state_dict(wL)
    final_losses = compute_loss_for_canaries()
    scores = [initial - final for initial, final in zip(initial_losses, final_losses)]
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

    # TODO recap on all params...

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
    parser.add_argument('--flip_canaries', type=bool, default=False, help='whether to randomly flip canaries')
    parser.add_argument('--n', type=int, default=0, help='number of non-auditing samples')
    args = parser.parse_args()

    # Run main function
    main(args)
