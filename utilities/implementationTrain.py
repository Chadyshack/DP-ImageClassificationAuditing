# Imports and configuration
import argparse
import os
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

# TODO FIX THESE COMMENTS
# Main function for training and auditing
def main(args):

    # Set torch device
    device=torch.device("cuda:0")

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
    }

    # Load test and train datasets (denoting as full because this is reduced later)
    data_dir = '/s/lovelace/c/nobackup/iray/dp-imgclass/PediatricChestX-rayPneumoniaData'
    full_trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])

    # Set up train loader (also thought of as x_IN loader) and the test loader, no x_OUT loader needed
    trainloader = torch.utils.data.DataLoader(full_trainset, batch_size=args.mini_bs, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    # Handle for gradient accumulation steps
    n_acc_steps = args.bs // args.mini_bs

    # Creae model and validate
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
                                     sample_rate = args.bs / len(full_trainset),
                                     epochs = args.epochs)
        print(f'adding noise level {sigma}')

        # Build privacy engine and attach it to the current optimizer
        privacy_engine = PrivacyEngine(
            net,
            batch_size=args.bs,
            sample_size=len(full_trainset),
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

    # Run train and test functions for number of epochs, but save initial state of model first
    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)

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
    args = parser.parse_args()

    # Run main function
    main(args)
