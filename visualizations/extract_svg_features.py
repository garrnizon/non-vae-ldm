"""
Extract SVG encoder features from images in imagenet-10 folder.
This script loads the DINOv3 encoder and extracts features from images.
"""

import sys
import os
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add paths
sys.path.append('SVG/')
sys.path.append('SVG/autoencoder')

# Check if we have a DINOv3 checkpoint
DINOV3_CHECKPOINT = "dinov3_vits16plus_pretrain_lvd1689m.pth"
if os.path.exists(DINOV3_CHECKPOINT):
    print(f"Found DINOv3 checkpoint: {DINOV3_CHECKPOINT}")
    dinov3_path = DINOV3_CHECKPOINT
else:
    print(f"Warning: {DINOV3_CHECKPOINT} not found. Will try to load from torch.hub")
    dinov3_path = None

def load_dinov3_encoder(dinov3_path=None, dinov3_repo_path=None):
    """
    Load DINOv3 encoder from checkpoint.
    DINOv3 ViT-S/16+ model loading.
    
    Args:
        dinov3_path: Path to DINOv3 checkpoint (.pth file)
        dinov3_repo_path: Path to cloned DINOv3 repository (optional)
                          If None, will try to find it automatically
    """
    if not dinov3_path or not os.path.exists(dinov3_path):
        raise FileNotFoundError(
            f"DINOv3 checkpoint not found: {dinov3_path}\n"
            f"Please ensure the checkpoint file exists.\n"
            f"Expected file: dinov3_vits16plus_pretrain_lvd1689m.pth"
        )
    
    print(f"Loading DINOv3 from checkpoint: {dinov3_path}")
    
    # Load checkpoint (it's a direct OrderedDict)
    print(f"Loading checkpoint...")
    checkpoint = torch.load(dinov3_path, map_location='cpu', weights_only=False)
    
    # The checkpoint is a direct state_dict (OrderedDict)
    state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Try to find DINOv3 repository
    if dinov3_repo_path is None:
        # Check common locations
        possible_paths = [
            'dinov3',
            '../dinov3',
            '../../dinov3',
            os.path.expanduser('~/dinov3'),
        ]
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                dinov3_repo_path = os.path.abspath(path)
                print(f"Found DINOv3 repository at: {dinov3_repo_path}")
                break
    
    # Load DINOv3 architecture
    if dinov3_repo_path and os.path.exists(dinov3_repo_path):
        # Load from local DINOv3 repository (preferred method)
        print(f"Loading DINOv3 architecture from local repository: {dinov3_repo_path}")
        try:
            model = torch.hub.load(
                repo_or_dir=dinov3_repo_path,
                model='dinov3_vits16plus',
                source="local",
                pretrained=False,  # We'll load weights separately
            )
            print("DINOv3 architecture loaded from local repository")
            
            # Load checkpoint weights
            model.load_state_dict(state_dict, strict=False)
            print("✓ DINOv3 weights loaded successfully!")
            
        except Exception as e:
            print(f"Error loading from local repo: {e}")
            raise
    else:
        # DINOv3 requires the official repository
        raise FileNotFoundError(
            "DINOv3 repository not found!\n"
            "Please clone the DINOv3 repository:\n"
            "  git clone https://github.com/facebookresearch/dinov3.git\n"
            "Then either:\n"
            "  1. Place it in the current directory as 'dinov3/', or\n"
            "  2. Pass dinov3_repo_path parameter to load_dinov3_encoder()"
        )
    
    model.eval()
    return model

def preprocess_image(image_path, size=256):
    """
    Preprocess image for DINOv3 encoder.
    Same preprocessing as used in DinoDecoder.get_input()
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Resize to 256x256 (center crop if needed)
    img = img.resize((size, size), Image.BICUBIC)
    
    # Convert to tensor and normalize to [0, 1]
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
    
    # Normalize with ImageNet stats (same as DinoDecoder)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor.unsqueeze(0)  # Add batch dimension

def extract_features(encoder, image_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Extract features from a single image using DINOv3 encoder.
    Returns features in the same format as DinoDecoder.encode()
    """
    encoder = encoder.to(device)
    
    # Preprocess image
    img_tensor = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # Extract features
    with torch.no_grad():
        # DINOv3 forward_features returns a dict with 'x_norm_patchtokens'
        features = encoder.forward_features(img_tensor)
        
        # Get patch tokens (same as in DinoDecoder.encode())
        if isinstance(features, dict):
            if 'x_norm_patchtokens' in features:
                h = features['x_norm_patchtokens']  # [B, N, D]
            elif 'x_prenorm' in features:
                h = features['x_prenorm']
            else:
                # Fallback: use the first tensor value
                h = list(features.values())[0]
        else:
            h = features
        
        # Convert to [B, D, N] format (if needed)
        if h.dim() == 3 and h.shape[1] > h.shape[2]:
            # Already in [B, N, D] format, transpose to [B, D, N]
            h = h.permute(0, 2, 1)
        
        # Reshape to [B, D, H_patch, W_patch] format
        B, D, N = h.shape
        H_patch = W_patch = int(np.sqrt(N))
        h = h.view(B, D, H_patch, W_patch)
    
    return h.cpu()

def process_folder(folder_path, encoder, output_dir=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Process all images in a folder and extract features.
    """
    folder_path = Path(folder_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}
    
    # Find all images
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(folder_path.rglob(f'*{ext}'))
    
    print(f"Found {len(image_paths)} images in {folder_path}")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    features_dict = {}
    for img_path in tqdm(image_paths, desc="Extracting features"):
        try:
            features = extract_features(encoder, str(img_path), device)
            rel_path = img_path.relative_to(folder_path)
            features_dict[str(rel_path)] = features.squeeze(0)  # Remove batch dimension
            
            # Optionally save individual feature files
            if output_dir:
                feature_path = Path(output_dir) / f"{rel_path.stem}.pt"
                feature_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(features.squeeze(0), feature_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return features_dict

def main():
    # Configuration
    image_folder = "imagenet-10"
    output_dir = "svg_features"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Try to find DINOv3 checkpoint
    dinov3_checkpoint = "dinov3_vits16plus_pretrain_lvd1689m.pth"
    if not os.path.exists(dinov3_checkpoint):
        # Try alternative names
        alt_names = [
            "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
            "dinov3_vits16plus_pretrain.pth",
        ]
        for alt_name in alt_names:
            if os.path.exists(alt_name):
                dinov3_checkpoint = alt_name
                break
        else:
            dinov3_checkpoint = None
    
    # Load DINOv3 encoder
    print("Loading DINOv3 encoder...")
    try:
        encoder = load_dinov3_encoder(dinov3_path=dinov3_checkpoint)
    except FileNotFoundError as e:
        print("\n" + "="*60)
        print("SETUP REQUIRED:")
        print("="*60)
        print(str(e))
        print("="*60)
        sys.exit(1)
    
    # Process images
    print(f"\nProcessing images from {image_folder}...")
    features_dict = process_folder(image_folder, encoder, output_dir, device)
    
    # Save all features
    output_file = "svg_features_all.pt"
    print(f"\nSaving all features to {output_file}...")
    torch.save(features_dict, output_file)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  - Processed {len(features_dict)} images")
    if features_dict:
        sample_feat = list(features_dict.values())[0]
        print(f"  - Feature shape: {sample_feat.shape}")
        print(f"  - Feature dtype: {sample_feat.dtype}")
    
    print(f"\nFeatures saved to:")
    print(f"  - Individual files: {output_dir}/")
    print(f"  - Combined file: {output_file}")

if __name__ == "__main__":
    main()
