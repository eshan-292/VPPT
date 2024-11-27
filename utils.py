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
import tarfile



# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Download and extract Tiny ImageNet
def download_and_extract_tiny_imagenet(destination_folder='./data'):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    filepath = os.path.join(destination_folder, filename)
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    if not os.path.exists(filepath):
        print("Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")
    else:
        print("Tiny ImageNet already downloaded.")
    
    # Extract the zip file
    extracted_folder = os.path.join(destination_folder, 'tiny-imagenet-200')
    if not os.path.exists(extracted_folder):
        print("Extracting Tiny ImageNet...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)
        print("Extraction complete.")
    else:
        print("Tiny ImageNet already extracted.")
    
    return extracted_folder



# Function to reorganize Tiny ImageNet validation set
def reorganize_tiny_val(tiny_val_dataset, tiny_imagenet_path):
    image_dir = os.path.join(os.path.join(tiny_imagenet_path, 'val'), 'images')
    val_dir = os.path.join(tiny_imagenet_path, 'val')
    annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    with open(annotations_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split('\t')
        img_name = parts[0]
        class_name = parts[1]
        # Create class folder if it doesn't exist
        class_folder = os.path.join(val_dir, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        # Move image to class folder
        src = os.path.join(image_dir, img_name)
        dst = os.path.join(class_folder, img_name)
        if not os.path.exists(dst):
            os.rename(src, dst)
    
    # After moving, remove images from 'val' root
    for img in os.listdir(image_dir):
        if os.path.isfile(os.path.join(image_dir, img)):
            os.remove(os.path.join(image_dir, img))
    
    print("Reorganized Tiny ImageNet validation set.")


# Function to create few-shot subsets
def get_few_shot_subset(dataset, num_shots=5, seed=42):
    # Ensure reproducibility
    set_seed(seed)
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    indices = []
    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        selected = np.random.choice(cls_indices, num_shots, replace=False)
        indices.extend(selected)
    return Subset(dataset, indices)

# Function to create data loaders
def get_data_loaders(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)



# Evaluation function
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(test_loader, desc='Evaluating')
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100. * correct / total
    print(f'Test Accuracy: {acc:.2f}%')
    return acc


# Define optimizers (only for prompts and classifier)
def get_optimizer(model, vppt=False ,lr=1e-3):
    if vppt:
        return optim.Adam([
            {'params': model.vpt.prompts},
            {'params': model.vpt.classifier.parameters()}
        ], lr=lr)
    
    return optim.Adam([
        {'params': model.prompts},
        {'params': model.classifier.parameters()}
    ], lr=lr)





def download_and_extract_oxford_flowers(destination_folder='./data/oxford_flowers'):
    url_images = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
    url_labels = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
    url_test = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat'
    
    import scipy.io
    
    filenames = ['102flowers.tgz', 'imagelabels.mat', 'setid.mat']
    urls = [url_images, url_labels, url_test]
    
    for filename, url in zip(filenames, urls):
        filepath = os.path.join(destination_folder, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded {filename}.")
        else:
            print(f"{filename} already downloaded.")
    
    # Extract images
    images_dir = os.path.join(destination_folder, 'jpg')
    if not os.path.exists(images_dir):
        print("Extracting Oxford Flowers images...")
        with tarfile.open(os.path.join(destination_folder, '102flowers.tgz'), 'r:gz') as tar:
            tar.extractall(path=destination_folder)
        print("Extraction complete.")
    else:
        print("Oxford Flowers images already extracted.")
    
    return destination_folder
def reorganize_oxford_flowers(oxford_path):
    import scipy.io
    import shutil
    
    labels_file = os.path.join(oxford_path, 'imagelabels.mat')
    setid_file = os.path.join(oxford_path, 'setid.mat')
    images_dir = os.path.join(oxford_path, 'jpg')
    
    # Load labels and set IDs
    labels = scipy.io.loadmat(labels_file)['labels'][0]
    setid = scipy.io.loadmat(setid_file)
    train_ids = setid['trnid'][0]
    val_ids = setid['valid'][0]
    test_ids = setid['tstid'][0]
    
    # Create train and val directories
    train_dir = os.path.join(oxford_path, 'train')
    val_dir = os.path.join(oxford_path, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    for img_id in train_ids:
        class_id = labels[img_id - 1]
        img_filename = f'image_{img_id:05d}.jpg'
        source = os.path.join(images_dir, img_filename)
        dest_class_dir = os.path.join(train_dir, f'class_{class_id}')
        os.makedirs(dest_class_dir, exist_ok=True)
        dest = os.path.join(dest_class_dir, img_filename)
        shutil.copy(source, dest)
    
    for img_id in val_ids:
        class_id = labels[img_id - 1]
        img_filename = f'image_{img_id:05d}.jpg'
        source = os.path.join(images_dir, img_filename)
        dest_class_dir = os.path.join(val_dir, f'class_{class_id}')
        os.makedirs(dest_class_dir, exist_ok=True)
        dest = os.path.join(dest_class_dir, img_filename)
        shutil.copy(source, dest)
    
    print("Reorganized Oxford Flowers into train and val folders.")





# main function
if __name__ == '__main__':
    # Set seed for reproducibility
    seed = 42
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Download and extract Oxford Flowers
    oxford_flowers_path = download_and_extract_oxford_flowers()

    # Reorganize Oxford Flowers dataset
    reorganize_oxford_flowers(oxford_flowers_path)
    

