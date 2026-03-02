import yaml
import json
import os
import sys


# === Load config ===
with open("configs/interpolation_sanity_check_base.yaml", "r") as f:
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

def plot_norms(tensor_list, save_path):
    size          = tensor_list[0].shape[0]
    num_timesteps = len(tensor_list)
    timesteps     = list(range(num_timesteps))

    COLORS = [
        "#2563EB", "#7C3AED", "#DB2777", "#D97706", "#059669",
        "#0891B2", "#DC2626", "#65A30D", "#EA580C", "#6D28D9"
    ]

    # ── Plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    for i in range(size):
        values = [tensor_list[t][i].item() for t in range(num_timesteps)]
        ax.plot(timesteps, values,
                color=COLORS[i % len(COLORS)],
                linewidth=1.8, marker="o", markersize=4,
                label=f"Element {i}")

    ax.set_title("Tensor Elements Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # Legend placed outside the plot to avoid overlap
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        fontsize=9,
        title="Elements"
    )

    # size         = tensor_list[0].shape[0]   # 10
    # num_timesteps = len(tensor_list)
    # timesteps    = list(range(num_timesteps))

    # COLORS = [
    #     "#2563EB", "#7C3AED", "#DB2777", "#D97706", "#059669",
    #     "#0891B2", "#DC2626", "#65A30D", "#EA580C", "#6D28D9"
    # ]

    # # ── Layout ───────────────────────────────────────────────────
    # cols = 5
    # rows = -(-size // cols)   # ceiling division

    # fig, axes = plt.subplots(
    #     rows, cols,
    #     figsize=(cols * 3.6, rows * 3.2),
    #     sharex=True,
    #     sharey=True,
    # )
    # fig.suptitle("Tensor Elements Over Time", fontsize=14, fontweight="bold", y=1.01)

    # for i in range(size):
    #     r, c = i // cols, i % cols
    #     ax = axes[r][c]

    #     # Extract values: works for CPU/GPU tensors
    #     values = [tensor_list[t][i].item() for t in range(num_timesteps)]

    #     ax.plot(timesteps, values,
    #             color=COLORS[i % len(COLORS)],
    #             linewidth=1.8, marker="o", markersize=3, label=f"Elem {i}")
    #     ax.fill_between(timesteps, values, alpha=0.08, color=COLORS[i % len(COLORS)])
    #     ax.set_title(f"Element {i}", fontsize=10, pad=4)
    #     ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    #     ax.tick_params(labelsize=8)

    #     if c == 0:
    #         ax.set_ylabel("Value", fontsize=9)
    #     if r == rows - 1:
    #         ax.set_xlabel("Timestep", fontsize=9)

    # # Hide any unused subplots (if size < rows*cols)
    # for j in range(size, rows * cols):
    #     axes[j // cols][j % cols].set_visible(False)

    # plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────
    OUTPUT_PATH = save_path.replace('.png', '_norms.png')
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved → {OUTPUT_PATH}")


def run_pipeline(
    z_interp: torch.Tensor,
    class_label: int,
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
    label_name = idx_to_label[class_label]
    save_path = (
        f"{save_dir}/{class_label}_{label_name}_{seed}_{suffix}.png"
    )
    y      = torch.tensor([class_label], device=device)
    y_null = torch.full_like(y, null_class_label)

    samples = diffusion.sample(
        z_interp,
        cond=y,
        null_cond=y_null,
        sample_steps=num_steps,
        cfg=cfg_scale,
        mode=mode,
        timestep_shift=timestep_shift,
        cfg_mode=cfg_mode,
        experiment=exp_name
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
    print(f"[{suffix}] Saving to {save_path}")
    torch.save(decoded, f'{save_path.replace(".png", "_torch.pt")}', pickle_protocol=4)

    return decoded

SIZE=1024
z_batch = torch.randn(SIZE, (image_size // 16) ** 2, z_channels, device=device)
class_label = class_labels[0]
label_name = idx_to_label[class_label]
print(class_label, label_name)
BATCH = cfg['batch_size']

from tqdm import trange

for i in trange(0, SIZE, BATCH):
    z = z_batch[i : i + BATCH]

    decoded_l = run_pipeline(z, class_label, suffix=f'{i}-{i+BATCH}')

print("All done.")
