
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

from vppt import VPPTViT, VPPTViT_Pretrain



# Define loss function
criterion = nn.CrossEntropyLoss()




# Pretraining function
def pretrain_prompts(model, full_train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(full_train_loader, desc=f'Pretraining Epoch {epoch+1}/{epochs}')
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
        
        epoch_loss = running_loss / len(full_train_loader.dataset)
        epoch_acc = 100. * correct / total
        print(f'Pretraining Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Training Accuracy: {epoch_acc:.2f}%')
        # save the prompts after every 10 epochs
        if (epoch+1)%5 == 0:
            prompts = model.get_prompts().detach().clone()
            torch.save(prompts, f'pretrained_prompts_{epoch+1}.pth')
            print(f"Pretrained prompts saved successfully for epoch {epoch+1}.")




# Main execution function
def main():
    # Set seed for reproducibility
    seed = 42
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Download and extract Tiny ImageNet
    tiny_imagenet_path = download_and_extract_tiny_imagenet()
    
    # Load Tiny ImageNet dataset
    transform_tiny = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),  # Using ImageNet's mean and std
                             std=(0.229, 0.224, 0.225))
    ])
    
    tiny_train_dataset = datasets.ImageFolder(root=os.path.join(tiny_imagenet_path, 'train'), transform=transform_tiny)
    tiny_val_dataset = datasets.ImageFolder(root=os.path.join(tiny_imagenet_path, 'val'), transform=transform_tiny)
    
    # Reorganize validation set
    # reorganize_tiny_val(tiny_val_dataset, tiny_imagenet_path)
    
    # Reload the validation dataset after reorganizing
    tiny_val_dataset = datasets.ImageFolder(root=os.path.join(tiny_imagenet_path, 'val'), transform=transform_tiny)
    
    # Data loaders for Tiny ImageNet
    tiny_train_loader = DataLoader(tiny_train_dataset, batch_size=128, shuffle=True, num_workers=4)
    tiny_val_loader = DataLoader(tiny_val_dataset, batch_size=128, shuffle=False, num_workers=4)
    

    # Initialize pretraining models
    pretrain_vppt = VPPTViT_Pretrain(prompt_length=100, num_classes=200).to(device)

    # Define optimizers for pretraining models
    pretrained_optimizer = get_optimizer(pretrain_vppt, vppt=True)
    
    
    # Pretraining VPPT models on full Tiny ImageNet training set
    print("\nPretraining VPPT model on full Tiny ImageNet training set...")
    pretrain_prompts(pretrain_vppt, tiny_train_loader, criterion, pretrained_optimizer, device, epochs=100)

    pretrained_prompts = pretrain_vppt.get_prompts().detach().clone()
    
    # save the pretrained_prompts
    torch.save(pretrained_prompts, 'pretrained_prompts.pth')
    print("Pretrained prompts saved successfully.")

    # eval on the val set
    
    

if __name__ == "__main__":
    main()
