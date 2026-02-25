import json
import os
import sys


# === Load config ===
with open("configs/interpolation_sanity_check_base.json", "r") as f:
    cfg = json.load(f)

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
from interpolations import interpolate_linear, interpolate_slerp


torch.set_grad_enabled(False)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# === Paths & experiment name ===
exp_name    = cfg["exp_name"]
ckpt_path   = cfg["ckpt_path"]
config_path = cfg["config_path"]

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

y      = torch.tensor(class_labels, device=device)
y_null = torch.full_like(y, null_class_label)

diffusion = RectifiedFlow(model)
dinov3    = dinov3.to(device)


# =========================================================
# === Pipeline: denoise → normalize → decode → save     ===
# =========================================================

def run_pipeline(
    z_interp: torch.Tensor,
    suffix: str,
) -> torch.Tensor:
    """
    Full pipeline for one interpolation sample:
      1. Denoise with RectifiedFlow
      2. Optionally apply feature normalization
      3. Reshape latents to spatial grid
      4. Decode with DINOv3 decoder
      5. Clamp output to [-1, 1]
      6. Save image and tensor to disk

    Args:
        z_interp: Interpolated latent tensor of shape (B, T, D)
        suffix:   Short string tag appended to output filenames (e.g. "s" or "l")

    Returns:
        decoded: Decoded image tensor of shape (B, C, H, W), clamped to [-1, 1]
    """
    # 1. Denoise
    print(f"[{suffix}] Sampling with cfg_mode={cfg_mode}, steps={num_steps}, cfg={cfg_scale}")
    samples = diffusion.sample(
        z_interp,
        cond=y,
        null_cond=y_null,
        sample_steps=num_steps,
        cfg=cfg_scale,
        mode=mode,
        timestep_shift=timestep_shift,
        cfg_mode=cfg_mode,
    )

    # 2. Feature normalization
    if config.basic.get("feature_norm", False):
        samples = samples * sp_std + sp_mean

    # 3. Reshape [B, T, D] -> [B, D, H, W]
    B, T, D = samples.shape
    lat_h = image_size // 16
    samples_latent = samples.permute(0, 2, 1).reshape(B, D, lat_h, lat_h)

    # 4. Decode
    print(f"[{suffix}] Decoding latents")
    with torch.no_grad():
        decoded = dinov3.decode(samples_latent)

    # 5. Clamp
    decoded = torch.clamp(decoded, -1, 1)

    # 6. Save
    save_path = (
        f"{cfg_mode}_sample_{exp_name}_"
        f"steps{num_steps}_{mode}_cfg{cfg_scale}_shift{timestep_shift}_{image_size}_{suffix}.png"
    )
    print(f"[{suffix}] Saving to {save_path}")
    save_image(decoded, f"res/{save_path}", nrow=decoded.shape[0], normalize=True, value_range=(-1, 1))
    torch.save(decoded, f'res/torch_{save_path.replace(".png", ".pt")}', pickle_protocol=4)

    return decoded

# === Interpolation steps ===
interp_steps  = cfg["interpolation_steps"]
slerp_dot_thr = cfg["slerp_dot_thr"]

z = torch.randn(2, (image_size // 16) ** 2, z_channels, device=device)
print("initialized ends for following interpolations:", z.shape)
z_l = interpolate_slerp(z, steps=interp_steps, dot_thr=slerp_dot_thr)
z_s = interpolate_linear(z, steps=interp_steps)
print("got interpolants")
print(z_l.shape, z_s.shape)

# === Run pipelines ===
decoded_l = run_pipeline(z_l, suffix="l")
decoded_s = run_pipeline(z_s, suffix="s")

print("All done.")
