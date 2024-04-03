# Imports and configuration
import argparse
import os
import torch.utils
import torch.utils.data
from fastDP import PrivacyEngine
import torch
import torchvision
torch.manual_seed(2)
import torch.nn as nn
import torch.optim as optim
import timm
from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier
from tqdm import tqdm
import numpy as np
import warnings; warnings.filterwarnings("ignore")

# Main function for training and auditing
def main(args):

    # Set torch device
    device= torch.device("cuda:0")

    # Data directory and transformations
    data_dir = '/s/lovelace/c/nobackup/iray/dp-imgclass/PediatricChestX-rayPneumoniaData'
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

    # Load test and train datasets
    trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
    
    # Set up train and test loader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.mini_bs, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    # Handle for gradient accumulation steps
    n_acc_steps = args.bs // args.mini_bs

    # Creae model and validate
    net = timm.create_model(args.model, pretrained = True, num_classes = 2)
    net = ModuleValidator.fix(net).to(device)

    # Create optimizer and loss functions
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

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

    # Privacy engine
    if 'nonDP' not in args.clipping_mode:
        # Calculate and display the noise level
        sigma = get_noise_multiplier(target_epsilon = args.epsilon,
                                     target_delta = 1 / len(trainset),
                                     sample_rate = args.bs / len(trainset),
                                     epochs = args.epochs)
        print(f'adding noise level {sigma}')
        
        # Build privacy engine and attach it to the current optimizer
        privacy_engine = PrivacyEngine(
            net,
            batch_size=args.bs,
            sample_size=len(trainset),
            noise_multiplier=sigma,
            epochs=args.epochs,
            clipping_mode='MixOpt',
            clipping_style='all-layer',
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
            te_acc.append(100. * correct/total)
            print('Epoch: ', epoch, 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Run train and test functions for number of epochs
    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)

    # Print final metrics
    print(tr_loss,tr_acc,te_loss,te_acc)


if __name__ == '__main__':
    # Create and parse arguments
    parser = argparse.ArgumentParser(description='Image Classification and Auditing')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--epochs', default=5, type=int, help='numter of epochs')
    parser.add_argument('--bs', default=100, type=int, help='batch size')
    parser.add_argument('--mini_bs', type=int, default=100)
    parser.add_argument('--epsilon', default=2, type=float, help='target epsilon')
    parser.add_argument('--dataset_name', type=str, default='NOT_USED', help='https://pytorch.org/vision/stable/datasets.html')
    parser.add_argument('--clipping_mode', type=str, default='MixOpt', choices=['BiTFiT','MixOpt', 'nonDP','nonDP-BiTFiT'])
    parser.add_argument('--model', default='beit_base_patch16_224.in22k_ft_in22k', type=str, help='model name')
    args = parser.parse_args()

    # Run main function
    main(args)
