import os
import shutil
import random
from pathlib import Path
from typing import List, Optional
import argparse

def create_imagenet_subset(
    source_path: str,
    output_path: str,
    num_classes: int = 100,
    num_train_per_class: int = 100,
    num_val_per_class: int = 50,
    seed: int = 42,
    class_list: Optional[List[str]] = None
):
    """
    Create a smaller ImageNet-style dataset subset.
    
    Args:
        source_path: Path to the original ImageNet dataset (must contain 'train' and 'val' folders)
        output_path: Path where the subset will be created
        num_classes: Number of classes to include (default: 100)
        num_train_per_class: Number of training images per class (default: 100)
        num_val_per_class: Number of validation images per class (default: 50)
        seed: Random seed for reproducibility (default: 42)
        class_list: Optional list of specific class names to use. If None, random classes are selected
    """
    
    random.seed(seed)
    
    source_train_dir = Path(source_path) / "train"
    source_val_dir = Path(source_path) / "val"
    
    # Verify source directories exist
    if not source_train_dir.exists():
        raise ValueError(f"Source training directory not found: {source_train_dir}")
    if not source_val_dir.exists():
        raise ValueError(f"Source validation directory not found: {source_val_dir}")
    
    # Get all available classes
    all_classes = sorted([d.name for d in source_train_dir.iterdir() if d.is_dir()])
    print(f"Found {len(all_classes)} classes in source dataset")
    
    # Select classes
    if class_list is not None:
        selected_classes = class_list[:num_classes]
        print(f"Using provided class list")
    else:
        selected_classes = random.sample(all_classes, min(num_classes, len(all_classes)))
        selected_classes.sort()
        print(f"Randomly selected {len(selected_classes)} classes")
    
    # Create output directories
    output_train_dir = Path(output_path) / "train"
    output_val_dir = Path(output_path) / "val"
    output_train_dir.mkdir(parents=True, exist_ok=True)
    output_val_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each class
    print("\nCreating subset dataset...")
    for idx, class_name in enumerate(selected_classes, 1):
        print(f"[{idx}/{len(selected_classes)}] Processing class: {class_name}")
        
        # Process training images
        source_class_train = source_train_dir / class_name
        output_class_train = output_train_dir / class_name
        output_class_train.mkdir(exist_ok=True)
        
        train_images = list(source_class_train.glob("*"))
        if len(train_images) < num_train_per_class:
            print(f"  Warning: Only {len(train_images)} training images available (requested {num_train_per_class})")
        
        selected_train = random.sample(train_images, min(num_train_per_class, len(train_images)))
        for img in selected_train:
            shutil.copy2(img, output_class_train / img.name)
        
        # Process validation images
        source_class_val = source_val_dir / class_name
        output_class_val = output_val_dir / class_name
        output_class_val.mkdir(exist_ok=True)
        
        val_images = list(source_class_val.glob("*"))
        if len(val_images) < num_val_per_class:
            print(f"  Warning: Only {len(val_images)} validation images available (requested {num_val_per_class})")
        
        selected_val = random.sample(val_images, min(num_val_per_class, len(val_images)))
        for img in selected_val:
            shutil.copy2(img, output_class_val / img.name)
        
        print(f"  Copied {len(selected_train)} train + {len(selected_val)} val images")
    
    # Save class list
    class_list_file = Path(output_path) / "classes.txt"
    with open(class_list_file, 'w') as f:
        for class_name in selected_classes:
            f.write(f"{class_name}\n")
    
    print(f"\n✓ Dataset created successfully at: {output_path}")
    print(f"  - {len(selected_classes)} classes")
    print(f"  - ~{num_train_per_class} training images per class")
    print(f"  - ~{num_val_per_class} validation images per class")
    print(f"  - Class list saved to: {class_list_file}")


def create_imagenet_subset_symlinks(
    source_path: str,
    output_path: str,
    num_classes: int = 100,
    seed: int = 42,
    class_list: Optional[List[str]] = None
):
    """
    Create a smaller ImageNet-style dataset subset using symbolic links (faster, no duplication).
    Note: This keeps ALL images from selected classes, not a limited number per class.
    
    Args:
        source_path: Path to the original ImageNet dataset
        output_path: Path where the subset will be created
        num_classes: Number of classes to include
        seed: Random seed for reproducibility
        class_list: Optional list of specific class names to use
    """
    
    random.seed(seed)
    
    source_train_dir = Path(source_path) / "train"
    source_val_dir = Path(source_path) / "val"
    
    # Get all available classes
    all_classes = sorted([d.name for d in source_train_dir.iterdir() if d.is_dir()])
    
    # Select classes
    if class_list is not None:
        selected_classes = class_list[:num_classes]
    else:
        selected_classes = random.sample(all_classes, min(num_classes, len(all_classes)))
        selected_classes.sort()
    
    # Create output directories
    output_train_dir = Path(output_path) / "train"
    output_val_dir = Path(output_path) / "val"
    output_train_dir.mkdir(parents=True, exist_ok=True)
    output_val_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating symlinks...")
    for class_name in selected_classes:
        # Create symlinks for train and val
        os.symlink(
            source_train_dir / class_name,
            output_train_dir / class_name
        )
        os.symlink(
            source_val_dir / class_name,
            output_val_dir / class_name
        )
    
    print(f"✓ Symlinked {len(selected_classes)} classes to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create a smaller ImageNet-style dataset (100 images per class for 100 classes)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--source-path', type=str, required=True,
                        help='Path to original ImageNet dataset (containing train/ and val/ folders)')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path where the subset will be created')
    parser.add_argument('--num-classes', type=int, default=100,
                        help='Number of classes to include')
    parser.add_argument('--num-train', type=int, default=100,
                        help='Number of training images per class')
    parser.add_argument('--num-val', type=int, default=50,
                        help='Number of validation images per class')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use-symlinks', action='store_true',
                        help='Use symlinks instead of copying (keeps all images from selected classes)')
    
    args = parser.parse_args()
    
    if args.use_symlinks:
        create_imagenet_subset_symlinks(
            source_path=args.source_path,
            output_path=args.output_path,
            num_classes=args.num_classes,
            seed=args.seed
        )
    else:
        create_imagenet_subset(
            source_path=args.source_path,
            output_path=args.output_path,
            num_classes=args.num_classes,
            num_train_per_class=args.num_train,
            num_val_per_class=args.num_val,
            seed=args.seed
        )
