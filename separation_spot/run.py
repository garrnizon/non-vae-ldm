import yaml
import json
import os
import sys
import sys

config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/separation_spot_base.yaml"

# === Load config ===
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = cfg["cuda_device"]

for p in cfg["sys_paths"]:
    sys.path.append(p)


import glob
import torch
from torchvision.utils import save_image
from PIL import Image
from IPython.display import display
from omegaconf import OmegaConf

from rectified_flow.rectified_flow import RectifiedFlow
from utils import instantiate_from_config
from utils import find_model


torch.set_grad_enabled(False)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# === Paths & experiment name ===
exp_name    = cfg["exp_name"]
ckpt_path   = cfg["ckpt_path"]
config_path = cfg["config_path"]
save_dir = cfg["save_dir"] + '/' + exp_name

os.makedirs(save_dir, exist_ok=True)


assert ckpt_path and os.path.exists(ckpt_path),    "Please set a valid ckpt_path"
assert config_path and os.path.exists(config_path), "Please set a valid config_path"
print(f"Experiment: {exp_name}")

config = OmegaConf.load(config_path)
print("config loaded")


# === Model ===
model = instantiate_from_config(config.model)
state_dict = find_model(ckpt_path)
model.load_state_dict(state_dict, strict=False)
model = model.to(device).eval()
print("model loaded")


# === Encoder ===
encoder_config = OmegaConf.load(config.basic.encoder_config)
dinov3 = instantiate_from_config(encoder_config.model).eval()
z_channels = encoder_config.model.params.ddconfig.z_channels
print("encoder loaded")


# === Globals ===
print("initializing globals")
seed            = cfg["seed"]
num_steps       = cfg["num_steps"]
cfg_scale       = cfg["cfg_scale"]
image_size      = cfg["image_size"]
samples_per_row = cfg["samples_per_row"]
class_labels    = cfg["class_labels"]

torch.manual_seed(seed)

# === Feature normalization stats ===
stats_path = cfg["stats_path"]
assert os.path.exists(stats_path), "Missing dinov3_sp_stats.pt"
stats   = torch.load(stats_path)
sp_mean = stats["dinov3_sp_mean"].to(device)[:, :, :z_channels]
sp_std  = stats["dinov3_sp_std"].to(device)[:, :, :z_channels]


# === Sampling settings ===
timestep_shift   = cfg["timestep_shift"]
cfg_mode         = cfg["cfg_mode"]
mode             = cfg["mode"]
null_class_label = cfg["null_class_label"]

diffusion = RectifiedFlow(model)
dinov3    = dinov3.to(device)

# === Initializing class labels ===
with open('data/imagenet_class_index.json', 'r') as f:
    class_index = json.load(f)

label_to_idx = {v[1]: int(k) for k, v in class_index.items()}

def get_index(label_str):
    results = []
    for part in label_str.split(','):
        part = '_'.join(part.strip().split())
        if part in label_to_idx:
            results.append(label_to_idx[part])
    return results

idx_to_label = {int(k): v[1] for k, v in class_index.items()}


# =========================================================
# === Pipeline: denoise → normalize → decode → save     ===
# =========================================================
import matplotlib.pyplot as plt


def run_pipeline(
    z_interp: torch.Tensor,
    suffix: str,
    steps: list[int],
    labels: list[int],
) -> torch.Tensor:
    # 1. Denoise
    filename = ' + '.join([
        f"{steps[s]}({labels[s]}, {label_names[s]})"
        for s in range(len(steps))
    ])
    print(f"processing steps: " + filename)

    save_path = (
        f"{save_dir}/{filename}.png"
    )
    y      = torch.tensor([0], device=device)
    y_null = torch.full_like(y, null_class_label)
    
    samples = z_interp.clone()

    labels = torch.tensor(labels, device=device)
    samples = diffusion.sample(
        samples,
        cond=None,
        null_cond=y_null,
        sample_steps=sum(steps),
        cfg=cfg_scale,
        mode=mode,
        timestep_shift=timestep_shift,
        cfg_mode=cfg_mode,
        experiment=exp_name,
        steps=steps,
        labels=labels,
    )

    # 2. Feature normalization
    if config.basic.get("feature_norm", False):
        samples = samples * sp_std + sp_mean

    # 3. Reshape [B, T, D] -> [B, D, H, W]
    B, T, D = samples.shape
    lat_h = image_size // 16
    samples_latent = samples.permute(0, 2, 1).reshape(B, D, lat_h, lat_h)

    # 4. Decode
    print(f"[{filename}] Decoding latents")
    with torch.no_grad():
        decoded = dinov3.decode(samples_latent)

    # 5. Clamp
    decoded = torch.clamp(decoded, -1, 1)

    # 6. Save
    print(f"[{filename}] Saving to {save_path}")
    save_image(decoded, f"{save_path}", nrow=decoded.shape[0], normalize=True, value_range=(-1, 1))
    torch.save(decoded, f'{save_path.replace(".png", "_torch.pt")}', pickle_protocol=4)

    return decoded

# steps = cfg["steps"]
labels = cfg["labels"]

z_batch = torch.randn(num_steps + 1, (image_size // 16) ** 2, z_channels, device=device)

for i in range(0, num_steps + 1):
    steps = [i, num_steps - i]
    
    label_names = [idx_to_label[class_label] for class_label in labels]

    z = z_batch[i:i+1]
    
    z_separated = run_pipeline(z, 'separated', steps, labels)

    print("done.")
# for i, class_label in enumerate(class_labels):
#     z = z_batch[i:i+1]

#     label_name = idx_to_label[class_label]


#     print("processing label", class_label, label_name)

#     z_at_once = run_pipeline(z, class_label, 'at_once', [sum(steps)], [class_label])
#     z_separated = run_pipeline(z, class_label, 'separated', steps, [class_label] * len(steps))

#     print(f'difference norm: {torch.sum((z_at_once - z_separated) ** 2)}')

#     print("done.")
