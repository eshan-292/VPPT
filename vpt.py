import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from transformers import ViTModel
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define the VPTViT model
class VPTViT(nn.Module):
    def __init__(self, vit_model_name='google/vit-base-patch16-224', prompt_length=10, num_classes=10):
        super(VPTViT, self).__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.vit.config.num_labels = num_classes
        
        # Freeze ViT parameters
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Initialize prompt tokens as learnable parameters
        self.prompt_length = prompt_length
        self.hidden_size = self.vit.config.hidden_size
        # Shape: (prompt_length, hidden_size)
        self.prompts = nn.Parameter(torch.randn(prompt_length, self.hidden_size))
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, num_classes)
    
    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        
        # Get input embeddings (including [CLS] token)
        embeddings = self.vit.embeddings(pixel_values=pixel_values)  # Shape: (batch_size, 1 + num_patches, hidden_size)
        
        # Expand prompts for the batch
        prompts = self.prompts.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, prompt_length, hidden_size)
        
        # Concatenate [CLS] token, prompts, and patch embeddings
        # Assuming [CLS] is the first token
        cls_token = embeddings[:, :1, :]  # Shape: (batch_size, 1, hidden_size)
        patch_embeddings = embeddings[:, 1:, :]  # Shape: (batch_size, num_patches, hidden_size)
        # ensure that they are on the same device
        prompts = prompts.to(cls_token.device)
        patch_embeddings = patch_embeddings.to(cls_token.device)
        
        combined_embeddings = torch.cat((cls_token, prompts, patch_embeddings), dim=1)  # Shape: (batch_size, 1 + prompt_length + num_patches, hidden_size)
        
        # Pass through transformer encoder
        encoder_outputs = self.vit.encoder(combined_embeddings)
        sequence_output = encoder_outputs.last_hidden_state  # Shape: (batch_size, 1 + prompt_length + num_patches, hidden_size)
        
        # Take the [CLS] token from the output
        cls_token_output = sequence_output[:, 0, :]  # Shape: (batch_size, hidden_size)
        
        # Classification
        logits = self.classifier(cls_token_output)  # Shape: (batch_size, num_classes)
        
        return logits