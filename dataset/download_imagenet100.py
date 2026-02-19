import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def export_split(ds_split, out_root: Path, label_names):
    out_root.mkdir(parents=True, exist_ok=True)

    for i, ex in enumerate(tqdm(ds_split, desc=f"Writing {out_root.name}")):
        img = ex["image"]
        label_id = int(ex["label"])
        class_name = label_names[label_id]

        class_dir = out_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # Save as JPEG (ImageNet-style). Filenames are deterministic.
        out_path = class_dir / f"{out_root.name}_{class_name}_{i:08d}.jpg"

        if isinstance(img, Image.Image):
            im = img.convert("RGB")
        else:
            # Sometimes datasets yields dict/array-like; try to coerce via PIL
            im = Image.fromarray(img).convert("RGB")

        im.save(out_path, format="JPEG", quality=95, optimize=True)


def main():
    target_dir = Path("./data/imagenet100").resolve()
    cache_dir = target_dir / "_hf_cache"   # keep HF cache inside your requested folder
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Downloads dataset artifacts into cache_dir, then loads them.
    ds = load_dataset("clane9/imagenet-100", cache_dir=str(cache_dir))

    # HF uses "validation" split name; we export it to an ImageNet-style "val" folder.
    label_names = ds["train"].features["label"].names

    export_split(ds["train"], target_dir / "train", label_names)
    export_split(ds["validation"], target_dir / "val", label_names)

    # Save classes.txt (one class name per line)
    with open(target_dir / "classes.txt", "w", encoding="utf-8") as f:
        for name in label_names:
            f.write(name + "\n")

    print(f"\nDone. Wrote ImageNet-style folders to: {target_dir}")
    print(f"HF cache kept in: {cache_dir}")

if __name__ == "__main__":
    main()
