
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from transformers import ViTModel, ViTConfig
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import random
import os
import zipfile
import urllib.request

from vpt import VPTViT
from utils import set_seed,download_and_extract_tiny_imagenet, reorganize_tiny_val, get_few_shot_subset, get_data_loaders, get_optimizer, evaluate




# Define the VPPTViT model
class VPPTViT(nn.Module):
    def __init__(self, vit_model_name='google/vit-base-patch16-224', prompt_length=10, num_classes=10):
        super(VPPTViT, self).__init__()
        self.vpt = VPTViT(vit_model_name, prompt_length, num_classes)
    
    def forward(self, pixel_values):
        return self.vpt(pixel_values)
    
    def get_prompts(self):
        return self.vpt.prompts
    
    def set_prompts(self, new_prompts):
        self.vpt.prompts = nn.Parameter(new_prompts)


# Define the VPPTViT_Pretrain model for pretraining
class VPPTViT_Pretrain(nn.Module):
    def __init__(self, vit_model_name='google/vit-base-patch16-224', prompt_length=10, num_classes=200):
        super(VPPTViT_Pretrain, self).__init__()
        self.vpt = VPTViT(vit_model_name, prompt_length, num_classes)
    
    def forward(self, pixel_values):
        return self.vpt(pixel_values)
    
    def get_prompts(self):
        return self.vpt.prompts
    
    def set_prompts(self, new_prompts):
        self.vpt.prompts = nn.Parameter(new_prompts)






# Initialize models
def initialize_models(device, num_classes=10):
    # Initialize VPT and VPPT models for CIFAR-10
    vpt_model_cifar10 = VPTViT(prompt_length=100, num_classes=num_classes).to(device)
    vppt_model_cifar10 = VPPTViT(prompt_length=100, num_classes=num_classes).to(device)
    
    # Initialize VPT and VPPT models for CIFAR-100
    vpt_model_cifar100 = VPTViT(prompt_length=100, num_classes=100).to(device)
    vppt_model_cifar100 = VPPTViT(prompt_length=100, num_classes=100).to(device)
    
    return vpt_model_cifar10, vppt_model_cifar10, vpt_model_cifar100, vppt_model_cifar100

# Define loss function
criterion = nn.CrossEntropyLoss()



# Fine-tuning function
def train_fine_tune(model, train_loader, criterion, optimizer, device, epochs=10, test_loader_cifar10=None):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f'Fine-Tuning Epoch {epoch+1}/{epochs}')
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            loop.set_postfix(loss=loss.item(), accuracy=100. * correct / total)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total
        print(f'Fine-Tuning Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Train Accuracy: {epoch_acc:.2f}%')
        # also print the test accuracy
        # test_acc = evaluate(model, test_loader_cifar10, device)
        # print(f'Fine-Tuning Epoch [{epoch+1}/{epochs}] Test Accuracy: {test_acc:.2f}%')



import matplotlib.pyplot as plt

# Updated fine-tuning function to record test accuracy at each epoch
def train_fine_tune_with_logging(model, train_loader, criterion, optimizer, device, epochs, test_loader, log_dict):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f'Fine-Tuning Epoch {epoch+1}/{epochs}')
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            loop.set_postfix(loss=loss.item(), accuracy=100. * correct / total)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total
        print(f'Fine-Tuning Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Train Accuracy: {epoch_acc:.2f}%')
        
        # Test accuracy for current epoch
        test_acc = evaluate(model, test_loader, device)
        print(f'Fine-Tuning Epoch [{epoch+1}/{epochs}] Test Accuracy: {test_acc:.2f}%')
        
        # Log test accuracy for convergence plot
        log_dict['test_accuracy'].append(test_acc)


# Plots the test accuracy of VPT and VPPT with different numbers of prompt tokens on CIFAR-100 under 4-shot settings.
def plot_accuracy_vs_prompt_tokens():
    
    # Set seed for reproducibility
    seed = 42
    set_seed(seed)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Define transformations for CIFAR-10 and CIFAR-100
    transform_few_shot = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),  # Normalize to [-1, 1]
                             std=(0.5, 0.5, 0.5))
    ])
    

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_few_shot)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_few_shot)
    
    
    epochs = 30
    batch_size = 128

    # Create few-shot subsets
    few_shot_train = get_few_shot_subset(train_dataset, num_shots=4, seed=seed)
    train_loader = get_data_loaders(few_shot_train, batch_size=batch_size, shuffle=True)
    test_loader = get_data_loaders(test_dataset, batch_size=batch_size, shuffle=False)



    
    
    log_dict_vpt = {'test_accuracy': []}
    log_dict_vppt = {'test_accuracy': []}
    for num_prompt_tokens in [1, 10,50,100]:
        print(f"\nTraining VPT with {num_prompt_tokens} prompt tokens...")
        vpt_model = VPTViT(prompt_length=num_prompt_tokens, num_classes=100).to(device)
        vpt_optimizer = get_optimizer(vpt_model)
        train_fine_tune(vpt_model, train_loader, criterion, vpt_optimizer, device, epochs, test_loader)
        vpt_acc = evaluate(vpt_model, test_loader, device)
        log_dict_vpt['test_accuracy'].append(vpt_acc)

        print(f"\nTraining VPPT with {num_prompt_tokens} prompt tokens...")
        vppt_model = VPPTViT(prompt_length=num_prompt_tokens, num_classes=100).to(device)
        pretrained_prompts = torch.load('pretrained_prompts.pth')
        # set only num_prompt_tokens tokens for the VPPT model from the pretrained prompts
        vppt_model.set_prompts(pretrained_prompts[:num_prompt_tokens])
        vppt_optimizer = get_optimizer(vppt_model, vppt=True)
        train_fine_tune(vppt_model, train_loader, criterion, vppt_optimizer, device, epochs, test_loader)
        vppt_acc = evaluate(vppt_model, test_loader, device)
        log_dict_vppt['test_accuracy'].append(vppt_acc)

    plt.figure(figsize=(10, 6))
    plt.plot([1, 10,50,100], log_dict_vpt['test_accuracy'], label='VPT')
    plt.plot([1, 10,50,100], log_dict_vppt['test_accuracy'], label='VPPT')
    plt.xlabel('Number of Prompt Tokens')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs. Number of Prompt Tokens (CIFAR-100, 4-Shot)')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_vs_prompt_tokens.png')
    plt.show()


        
        


# Main function to handle multiple few-shot settings and save results
def main():
    
    # plot_accuracy_vs_prompt_tokens()
    # exit()
    
    # Set seed for reproducibility
    seed = 42
    set_seed(seed)
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Define transformations for CIFAR-10 and CIFAR-100
    transform_few_shot = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),  # Normalize to [-1, 1]
                             std=(0.5, 0.5, 0.5))
    ])
    
    # Load the oxford_flowers dataset
    # train_dataset = datasets.Places365(root='./data', train=True, download=True, transform=transform_few_shot)
    # test_dataset = datasets.ImageNet(root='./data', train=False, download=True, transform=transform_few_shot)


    
    # train_dataset_flowers = datasets.ImageFolder(root='./data/oxford_flowers/train', transform=transform_few_shot)
    # test_dataset_flowers = datasets.ImageFolder(root='./data/oxford_flowers/val', transform=transform_few_shot)


    # Load CIFAR-10 and CIFAR-100 datasets
    train_dataset_cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_few_shot)
    test_dataset_cifar10 = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_few_shot)
    
    train_dataset_cifar100 = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_few_shot)
    test_dataset_cifar100 = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_few_shot)
    
    # Few-shot settings
    few_shot_settings = [1, 4, 16]  # Number of samples per class
    epochs = 30
    batch_size = 128


    
    # Open a file to store results
    with open('few_shot_results_dogs.txt', 'w') as result_file:
        result_file.write("Few-Shot Settings Results:\n")
        result_file.write("Shots, Dataset, Model, Test Accuracy\n")
        
        for num_shots in few_shot_settings:
            print(f"\nRunning few-shot experiment for num_shots={num_shots}...\n")
            
            # Create few-shot subsets
            few_shot_train_cifar10 = get_few_shot_subset(train_dataset_cifar10, num_shots=num_shots, seed=seed)
            few_shot_train_cifar100 = get_few_shot_subset(train_dataset_cifar100, num_shots=num_shots, seed=seed)

            # few_shot_train_flowers = get_few_shot_subset(train_dataset, num_shots=num_shots, seed=seed)
            
            # Data loaders
            train_loader_cifar10 = get_data_loaders(few_shot_train_cifar10, batch_size=batch_size, shuffle=True)
            test_loader_cifar10 = get_data_loaders(test_dataset_cifar10, batch_size=batch_size, shuffle=False)
            
            train_loader_cifar100 = get_data_loaders(few_shot_train_cifar100, batch_size=batch_size, shuffle=True)
            test_loader_cifar100 = get_data_loaders(test_dataset_cifar100, batch_size=batch_size, shuffle=False)

            # train_loader_flowers = get_data_loaders(few_shot_train_flowers, batch_size=batch_size, shuffle=True)
            # test_loader_flowers = get_data_loaders(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize models
            vpt_model_cifar10, vppt_model_cifar10, vpt_model_cifar100, vppt_model_cifar100 = initialize_models(device)
            # vpt_model, vppt_model, _, _ = initialize_models(device, num_classes=196)
            
            # Define optimizers
            vpt_optimizer_cifar10 = get_optimizer(vpt_model_cifar10)
            vppt_optimizer_cifar10 = get_optimizer(vppt_model_cifar10, vppt=True)
            # vpr_optimizer = get_optimizer(vpt_model)
            # vppt_optimizer = get_optimizer(vppt_model, vppt=True)
            
            vpt_optimizer_cifar100 = get_optimizer(vpt_model_cifar100)
            vppt_optimizer_cifar100 = get_optimizer(vppt_model_cifar100, vppt=True)

            # Transfer pretrained prompts to downstream VPPT models
            print("\nTransferring pretrained prompts to downstream VPPT models...")
            # load the saved pretrained prompts
            pretrained_prompts = torch.load('pretrained_prompts.pth')
            vppt_model_cifar10.set_prompts(pretrained_prompts)
            vppt_model_cifar100.set_prompts(pretrained_prompts)
            # vppt_model.set_prompts(pretrained_prompts)

            
            # For convergence plot (only for num_shots=1)
            log_dict_vpt_cifar10 = {'test_accuracy': []}
            log_dict_vppt_cifar10 = {'test_accuracy': []}
            log_dict_vpt_cifar100 = {'test_accuracy': []}
            log_dict_vppt_cifar100 = {'test_accuracy': []}


            log_dict_vpt = {'test_accuracy': []}
            log_dict_vppt = {'test_accuracy': []}
            

            if num_shots == 1:
                print("\nTraining VPT on CIFAR-100 (1-shot)...")
                # train_fine_tune_with_logging(vpt_model, train_loader_flowers, criterion, vpr_optimizer, device, epochs, test_loader_flowers, log_dict_vpt)
                train_fine_tune_with_logging(vpt_model_cifar100, train_loader_cifar100, criterion, vpt_optimizer_cifar100, device, epochs, test_loader_cifar100, log_dict_vpt_cifar100)
                

                
                print("\nTraining VPPT on CIFAR-100 (1-shot)...")
                # train_fine_tune_with_logging(vppt_model, train_loader_flowers, criterion, vppt_optimizer, device, epochs, test_loader_flowers, log_dict_vppt)
                train_fine_tune_with_logging(vppt_model_cifar100, train_loader_cifar100, criterion, vppt_optimizer_cifar100, device, epochs, test_loader_cifar100, log_dict_vppt_cifar100)
                
                
                # Plot convergence speed
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, epochs + 1), log_dict_vpt_cifar100['test_accuracy'], label='VPT')
                plt.plot(range(1, epochs + 1), log_dict_vppt_cifar100['test_accuracy'], label='VPPT')
                plt.xlabel('Epochs')
                plt.ylabel('Test Accuracy (%)')
                plt.title('Convergence Speed (CIFAR100, 1-Shot)')
                plt.legend()
                plt.grid(True)
                plt.savefig('convergence_cifar100_1_shot.png')
                plt.show()
            exit()
            
            # Train and evaluate for other settings
            print("\nTraining VPT on CIFAR-10...")
            # train_fine_tune(vpt_model_cifar10, train_loader_cifar10, criterion, vpt_optimizer_cifar10, device, epochs, test_loader_cifar10)
            train_fine_tune(vpt_model, train_loader_flowers, criterion, vpr_optimizer, device, epochs, test_loader_flowers)
            # vpt_acc_cifar10 = evaluate(vpt_model_cifar10, test_loader_cifar10, device)
            vpt_acc = evaluate(vpt_model, test_loader_flowers, device)

            
            print("\nTraining VPPT on CIFAR-10...")
            # train_fine_tune(vppt_model_cifar10, train_loader_cifar10, criterion, vppt_optimizer_cifar10, device, epochs, test_loader_cifar10)
            # vppt_acc_cifar10 = evaluate(vppt_model_cifar10, test_loader_cifar10, device)
            train_fine_tune(vppt_model, train_loader_flowers, criterion, vppt_optimizer, device, epochs, test_loader_flowers)
            vppt_acc = evaluate(vppt_model, test_loader_flowers, device)
            
            result_file.write(f"{num_shots}, Stanford Dogs, VPT, {vpt_acc:.2f}\n")
            result_file.write(f"{num_shots}, Stanford Dogs, VPPT, {vppt_acc:.2f}\n")

            # result_file.write(f"{num_shots}, CIFAR-10, VPT, {vpt_acc_cifar10:.2f}\n")
            # result_file.write(f"{num_shots}, CIFAR-10, VPPT, {vppt_acc_cifar10:.2f}\n")
            
            # print("\nTraining VPT on CIFAR-100...")
            # train_fine_tune(vpt_model_cifar100, train_loader_cifar100, criterion, vpt_optimizer_cifar100, device, epochs, test_loader_cifar100)
            # vpt_acc_cifar100 = evaluate(vpt_model_cifar100, test_loader_cifar100, device)
            
            # print("\nTraining VPPT on CIFAR-100...")
            # train_fine_tune(vppt_model_cifar100, train_loader_cifar100, criterion, vppt_optimizer_cifar100, device, epochs, test_loader_cifar100)
            # vppt_acc_cifar100 = evaluate(vppt_model_cifar100, test_loader_cifar100, device)
            
            # result_file.write(f"{num_shots}, CIFAR-100, VPT, {vpt_acc_cifar100:.2f}\n")
            # result_file.write(f"{num_shots}, CIFAR-100, VPPT, {vppt_acc_cifar100:.2f}\n")
        
        print("All experiments completed. Results saved to 'few_shot_results.txt'.")


if __name__ == "__main__":
    main()
