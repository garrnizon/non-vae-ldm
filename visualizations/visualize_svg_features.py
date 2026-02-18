"""
SVG Encoder Feature Visualization with t-SNE
Adapted from DINOv2 visualization to work with SVG encoder (DINOv3)
"""

import sys
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import random
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add paths for SVG modules
sys.path.append('SVG/')
sys.path.append('SVG/autoencoder')


class Config:
    """Configuration for SVG feature visualization"""
    # Model settings
    DINOV3_CHECKPOINT = 'dinov3_vits16plus_pretrain_lvd1689m.pth'
    DINOV3_REPO_PATH = 'dinov3'  # Path to cloned DINOv3 repository
    
    # Dataset settings
    NUM_CLASSES = 10
    SAMPLES_PER_CLASS = 100
    IMAGE_SIZE = 256  # SVG encoder uses 256x256
    
    # Feature aggregation
    FEATURE_AGGREGATION = 'global_avg'  # Options: 'global_avg', 'cls_token', 'flatten'
    
    # t-SNE settings
    TSNE_PERPLEXITY = 30
    TSNE_N_ITER = 1000
    TSNE_RANDOM_STATE = 42
    
    # Visualization settings
    FIGURE_SIZE = (16, 6)
    POINT_SIZE = 50
    POINT_ALPHA = 0.7
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Random seed
    SEED = 42


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dinov3_encoder(checkpoint_path, dinov3_repo_path=None):
    """
    Load DINOv3 encoder from checkpoint.
    """
    print(f"Loading DINOv3 from checkpoint: {checkpoint_path}")
    
    # Try to find DINOv3 repository
    if dinov3_repo_path is None:
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
    
    # Load checkpoint
    print(f"Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load DINOv3 architecture
    if dinov3_repo_path and os.path.exists(dinov3_repo_path):
        print(f"Loading DINOv3 architecture from local repository: {dinov3_repo_path}")
        encoder = torch.hub.load(
            repo_or_dir=dinov3_repo_path,
            model='dinov3_vits16plus',
            source="local",
            pretrained=False,
        )
        print("DINOv3 architecture loaded from local repository")
        
        # Load checkpoint weights
        encoder.load_state_dict(state_dict, strict=False)
        print("✓ DINOv3 weights loaded successfully!")
    else:
        raise FileNotFoundError(
            "DINOv3 repository not found!\n"
            "Please clone: git clone https://github.com/facebookresearch/dinov3.git"
        )
    
    encoder.eval()
    return encoder


class SVGFeatureExtractor:
    """Extracts features using SVG encoder (DINOv3)"""
    
    def __init__(self, checkpoint_path: str, dinov3_repo_path: str = None, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = load_dinov3_encoder(checkpoint_path, dinov3_repo_path)
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        print(f"SVG encoder loaded on {self.device}")
    
    def preprocess_image(self, image_path: str, size: int = 256):
        """Preprocess image for DINOv3 encoder (same as SVG preprocessing)"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((size, size), Image.BICUBIC)
        
        # Convert to tensor [0, 1]
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor.unsqueeze(0)  # Add batch dimension
    
    def extract_features(self, images: torch.Tensor, batch_size: int = 32, aggregation: str = 'global_avg') -> np.ndarray:
        """
        Extract features from images using SVG encoder
        
        Args:
            images: Tensor of shape (N, C, H, W) - already preprocessed
            batch_size: Batch size for processing
            aggregation: How to aggregate spatial features ('global_avg', 'cls_token', 'flatten')
            
        Returns:
            features: numpy array of shape (N, feature_dim)
        """
        features_list = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="Extracting SVG features"):
                batch = images[i:i+batch_size].to(self.device)
                
                # Extract features using DINOv3 encoder
                # This returns patch tokens: [B, N, D] or features dict
                features = self.encoder.forward_features(batch)
                
                # Handle different feature formats
                if isinstance(features, dict):
                    if 'x_norm_patchtokens' in features:
                        patch_tokens = features['x_norm_patchtokens']  # [B, N, D]
                    elif 'x_prenorm' in features:
                        patch_tokens = features['x_prenorm']
                    else:
                        patch_tokens = list(features.values())[0]
                else:
                    patch_tokens = features
                
                # Aggregate spatial features to get per-image features
                if aggregation == 'global_avg':
                    # Global average pooling over spatial dimensions
                    if patch_tokens.dim() == 3:  # [B, N, D]
                        img_features = patch_tokens.mean(dim=1)  # [B, D]
                    elif patch_tokens.dim() == 4:  # [B, D, H, W]
                        img_features = F.adaptive_avg_pool2d(patch_tokens, (1, 1)).squeeze(-1).squeeze(-1)  # [B, D]
                    else:
                        img_features = patch_tokens.mean()
                
                elif aggregation == 'cls_token':
                    # Use CLS token if available
                    if isinstance(features, dict) and 'x_norm_clstoken' in features:
                        img_features = features['x_norm_clstoken']  # [B, D]
                    else:
                        # Fallback to global avg
                        if patch_tokens.dim() == 3:
                            img_features = patch_tokens.mean(dim=1)
                        else:
                            img_features = F.adaptive_avg_pool2d(patch_tokens, (1, 1)).squeeze(-1).squeeze(-1)
                
                elif aggregation == 'flatten':
                    # Flatten spatial dimensions
                    if patch_tokens.dim() == 3:  # [B, N, D]
                        img_features = patch_tokens.view(patch_tokens.shape[0], -1)  # [B, N*D]
                    elif patch_tokens.dim() == 4:  # [B, D, H, W]
                        img_features = patch_tokens.view(patch_tokens.shape[0], -1)  # [B, D*H*W]
                    else:
                        img_features = patch_tokens.flatten(1)
                
                features_list.append(img_features.cpu().numpy())
        
        return np.concatenate(features_list, axis=0)


class ImageNetLoader:
    """Handles loading images from ImageNet folder structure"""
    
    def __init__(self, config: Config):
        self.config = config
        self.transform = self._get_transform()
    
    def _get_transform(self):
        """Get preprocessing transform for SVG encoder"""
        return transforms.Compose([
            transforms.Resize(self.config.IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def load_from_folder(self, root_path: str, class_folders: List[str], samples_per_class: int):
        """
        Load from local ImageNet folder structure
        Expected structure: root_path/class_folder/images.JPEG
        """
        images = []
        labels = []
        
        for class_idx, folder_name in enumerate(class_folders):
            class_path = os.path.join(root_path, folder_name)
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} not found")
                continue
            
            image_files = [
                f for f in os.listdir(class_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPEG'))
            ][:samples_per_class]
            
            for img_file in tqdm(image_files, desc=f"Loading {folder_name}"):
                img_path = os.path.join(class_path, img_file)
                img = Image.open(img_path).convert('RGB')
                images.append(self.transform(img))
                labels.append(class_idx)
        
        if len(images) == 0:
            return None, None
        
        return torch.stack(images), torch.tensor(labels)
    
    def load_synthetic_demo(self, num_classes: int, samples_per_class: int):
        """Create synthetic demo data using CIFAR-10"""
        from torchvision.datasets import CIFAR10
        
        print("Loading CIFAR-10 as demo dataset...")
        
        transform = transforms.Compose([
            transforms.Resize(self.config.IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        images = []
        labels = []
        
        cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']
        
        class_indices = {i: [] for i in range(min(num_classes, 10))}
        
        for idx, (_, label) in enumerate(tqdm(dataset, desc="Collecting indices")):
            if label < len(class_indices):
                class_indices[label].append(idx)
        
        for class_idx in tqdm(range(min(num_classes, 10)), desc="Sampling images"):
            indices = class_indices[class_idx]
            if indices:
                selected = random.sample(indices, min(samples_per_class, len(indices)))
                for idx in selected:
                    img, label = dataset[idx]
                    images.append(img)
                    labels.append(label)
        
        class_names = {i: (cifar_classes[i], i) for i in range(min(num_classes, 10))}
        
        return torch.stack(images), torch.tensor(labels), class_names


class FeatureVisualizer:
    """Handles t-SNE computation and visualization"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def compute_tsne(self, features: np.ndarray) -> np.ndarray:
        """Apply t-SNE dimensionality reduction"""
        print(f"Computing t-SNE on {features.shape[0]} samples with {features.shape[1]} dimensions...")
        
        tsne = TSNE(
            n_components=2,
            perplexity=min(self.config.TSNE_PERPLEXITY, features.shape[0] - 1),
            n_iter=self.config.TSNE_N_ITER,
            random_state=self.config.TSNE_RANDOM_STATE,
            verbose=1
        )
        
        embedded = tsne.fit_transform(features)
        print("t-SNE completed!")
        
        return embedded
    
    def plot_features_and_tsne(
        self,
        features: np.ndarray,
        tsne_embedded: np.ndarray,
        labels: np.ndarray,
        class_names: dict,
        save_path: Optional[str] = None
    ):
        """Create side-by-side visualization of feature space and t-SNE"""
        fig, axes = plt.subplots(1, 2, figsize=self.config.FIGURE_SIZE)
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        # Left plot: PCA of features
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features)
        
        ax1 = axes[0]
        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_name = class_names.get(label, (f'Class {label}', label))[0]
            ax1.scatter(
                features_pca[mask, 0],
                features_pca[mask, 1],
                c=[colors[i]],
                label=class_name,
                s=self.config.POINT_SIZE,
                alpha=self.config.POINT_ALPHA,
                edgecolors='white',
                linewidths=0.5
            )
        
        ax1.set_title('SVG Encoder Features (PCA)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('PC1', fontsize=12)
        ax1.set_ylabel('PC2', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Right plot: t-SNE embedding
        ax2 = axes[1]
        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_name = class_names.get(label, (f'Class {label}', label))[0]
            ax2.scatter(
                tsne_embedded[mask, 0],
                tsne_embedded[mask, 1],
                c=[colors[i]],
                label=class_name,
                s=self.config.POINT_SIZE,
                alpha=self.config.POINT_ALPHA,
                edgecolors='white',
                linewidths=0.5
            )
        
        ax2.set_title('t-SNE Visualization', fontsize=14, fontweight='bold')
        ax2.set_xlabel('t-SNE 1', fontsize=12)
        ax2.set_ylabel('t-SNE 2', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        ax2.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            fontsize=10,
            framealpha=0.9
        )
        
        plt.suptitle(
            'SVG Encoder Feature Space Semantic Separability',
            fontsize=16,
            fontweight='bold',
            y=1.02
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_tsne_only(
        self,
        tsne_embedded: np.ndarray,
        labels: np.ndarray,
        class_names: dict,
        save_path: Optional[str] = None
    ):
        """Create single t-SNE plot"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_name = class_names.get(label, (f'Class {label}', label))[0]
            ax.scatter(
                tsne_embedded[mask, 0],
                tsne_embedded[mask, 1],
                c=[colors[i]],
                label=class_name,
                s=self.config.POINT_SIZE * 1.5,
                alpha=self.config.POINT_ALPHA,
                edgecolors='white',
                linewidths=0.5
            )
        
        ax.set_title('SVG Encoder Features - t-SNE Visualization', fontsize=16, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        return fig


def run_visualization(
    use_demo_data: bool = True,
    imagenet_path: Optional[str] = None,
    num_classes: int = 10,
    samples_per_class: int = 100,
    save_path: str = 'svg_encoder_tsne_visualization.png',
    aggregation: str = 'global_avg'
):
    """
    Main function to run the complete visualization pipeline
    
    Args:
        use_demo_data: If True, use CIFAR-10 as demo. If False, load from imagenet_path
        imagenet_path: Path to local ImageNet folder (e.g., 'imagenet-10')
        num_classes: Number of classes to visualize
        samples_per_class: Number of samples per class
        save_path: Path to save the output figure
        aggregation: Feature aggregation method ('global_avg', 'cls_token', 'flatten')
    """
    config = Config()
    config.NUM_CLASSES = num_classes
    config.SAMPLES_PER_CLASS = samples_per_class
    config.FEATURE_AGGREGATION = aggregation
    
    set_seed(config.SEED)
    
    print("="*60)
    print("SVG Encoder Feature Visualization Pipeline")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    print(f"Checkpoint: {config.DINOV3_CHECKPOINT}")
    print(f"Classes: {num_classes}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Feature aggregation: {aggregation}")
    print("="*60)
    
    # Load dataset
    loader = ImageNetLoader(config)
    
    if use_demo_data:
        images, labels, class_names = loader.load_synthetic_demo(
            num_classes, samples_per_class
        )
    elif imagenet_path:
        # Load from imagenet-10 folder
        # Assuming single class folder structure
        class_folders = []
        for item in os.listdir(imagenet_path):
            item_path = os.path.join(imagenet_path, item)
            if os.path.isdir(item_path):
                class_folders.append(item)
        
        if len(class_folders) == 0:
            print(f"No class folders found in {imagenet_path}, using demo data")
            images, labels, class_names = loader.load_synthetic_demo(
                num_classes, samples_per_class
            )
        else:
            # Use first N folders as classes
            class_folders = class_folders[:num_classes]
            images, labels = loader.load_from_folder(
                imagenet_path, class_folders, samples_per_class
            )
            class_names = {i: (folder, i) for i, folder in enumerate(class_folders)}
            
            if images is None:
                print("Failed to load from folder, using demo data")
                images, labels, class_names = loader.load_synthetic_demo(
                    num_classes, samples_per_class
                )
    else:
        print("No data source specified, using demo data")
        images, labels, class_names = loader.load_synthetic_demo(
            num_classes, samples_per_class
        )
    
    print(f"\nDataset loaded: {len(images)} images")
    print(f"Classes: {list(class_names.values())}")
    
    # Extract features using SVG encoder
    extractor = SVGFeatureExtractor(
        config.DINOV3_CHECKPOINT,
        config.DINOV3_REPO_PATH,
        config.DEVICE
    )
    features = extractor.extract_features(
        images, 
        batch_size=32, 
        aggregation=aggregation
    )
    print(f"Features shape: {features.shape}")
    
    # Compute t-SNE
    visualizer = FeatureVisualizer(config)
    tsne_embedded = visualizer.compute_tsne(features)
    
    # Create visualization
    labels_np = labels.numpy()
    
    # Combined plot (PCA + t-SNE)
    visualizer.plot_features_and_tsne(
        features, tsne_embedded, labels_np, class_names,
        save_path=save_path
    )
    
    # t-SNE only plot
    tsne_save_path = save_path.replace('.png', '_tsne_only.png')
    visualizer.plot_tsne_only(
        tsne_embedded, labels_np, class_names,
        save_path=tsne_save_path
    )
    
    print("\nVisualization complete!")
    
    return features, tsne_embedded, labels_np, class_names


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize SVG encoder features with t-SNE')
    parser.add_argument('--imagenet_path', type=str, default=None,
                        help='Path to ImageNet folder (e.g., imagenet-10). If None, uses CIFAR-10 demo data')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes to visualize')
    parser.add_argument('--samples_per_class', type=int, default=100,
                        help='Number of samples per class')
    parser.add_argument('--save_path', type=str, default='svg_encoder_tsne_visualization.png',
                        help='Path to save the output figure')
    parser.add_argument('--aggregation', type=str, default='global_avg',
                        choices=['global_avg', 'cls_token', 'flatten'],
                        help='Feature aggregation method')
    parser.add_argument('--use_demo', action='store_true',
                        help='Force use of CIFAR-10 demo data')
    
    args = parser.parse_args()
    
    run_visualization(
        use_demo_data=args.use_demo or args.imagenet_path is None,
        imagenet_path=args.imagenet_path,
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        save_path=args.save_path,
        aggregation=args.aggregation
    )
