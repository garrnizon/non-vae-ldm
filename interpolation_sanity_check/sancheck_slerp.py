import json
import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import torch.multiprocessing as mp
from tqdm import trange
import time


with open("configs/base.json", "r") as f:
    cfg = json.load(f)

interp_steps  = cfg["interpolation_steps"]
slerp_dot_thr = cfg["slerp_dot_thr"]
image_size    = cfg["image_size"]


# ── Batched SLERP that RETURNS a hit-count instead of printing ──────────────
def interpolate_slerp_batched_counted(
    z: torch.Tensor,          # (B, 2, seq_len, z_channels)
    steps: int = 10,
    dot_thr: float = 0.9995,
) -> tuple[torch.Tensor, int]:
    """
    Returns:
        result  : (B, steps, seq_len, z_channels)
        n_hits  : number of pairs where |dot| > dot_thr  (the parallel branch)
    """
    B = z.shape[0]
    device, dtype = z.device, z.dtype

    z0 = z[:, 0].reshape(B, -1)   # (B, D)
    z1 = z[:, 1].reshape(B, -1)   # (B, D)

    z0_n = torch.nn.functional.normalize(z0, dim=1)   # (B, D)
    z1_n = torch.nn.functional.normalize(z1, dim=1)   # (B, D)

    dot = (z0_n * z1_n).sum(dim=1).clamp(-1, 1)       # (B,)
    parallel_mask = torch.abs(dot) > dot_thr           # (B,)  ← the branch you care about
    n_hits = int(parallel_mask.sum().item())

    theta     = torch.acos(dot)                        # (B,)
    sin_theta = torch.sin(theta)                       # (B,)

    ts = torch.linspace(0, 1, steps, device=device, dtype=dtype)   # (steps,)

    # Coefficients — shape (B, steps)
    safe_sin = sin_theta.clamp(min=1e-8).unsqueeze(1)  # avoid /0 in parallel branch
    s0 = torch.where(
        parallel_mask.unsqueeze(1),
        (1 - ts).unsqueeze(0).expand(B, -1),                          # linear fallback
        torch.sin(theta.unsqueeze(1) * (1 - ts).unsqueeze(0)) / safe_sin,
    )   # (B, steps)
    s1 = torch.where(
        parallel_mask.unsqueeze(1),
        ts.unsqueeze(0).expand(B, -1),
        torch.sin(theta.unsqueeze(1) * ts.unsqueeze(0)) / safe_sin,
    )   # (B, steps)

    # (B, steps, 1) * (B, 1, D) → (B, steps, D)
    result = s0.unsqueeze(2) * z0.unsqueeze(1) + s1.unsqueeze(2) * z1.unsqueeze(1)
    result = result.reshape(B, steps, *z.shape[2:])
    return result, n_hits


# ── Worker: runs a chunk of the outer loop and returns its hit count ─────────
def worker_fn(args):
    chunk_size, N, image_size, interp_steps, slerp_dot_thr = args
    device = torch.device("cuda:0")   # CUDA_VISIBLE_DEVICES="3" remaps to index 0
    seq_len = (image_size // 16) ** 2
    total_hits  = 0
    total_pairs = 0
    for _ in range(chunk_size):
        zs = torch.randn(N, 2, seq_len, 392, device=device)
        _, n_hits = interpolate_slerp_batched_counted(
            zs, steps=interp_steps, dot_thr=slerp_dot_thr
        )
        total_hits  += n_hits
        total_pairs += N
    return total_hits, total_pairs

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    start = time.time()
    times = 10 ** 4
    N     = 10 ** 2
    NUM_WORKERS = 8          # tune to your CPU core count; each shares GPU 3

    # Split outer iterations evenly across workers
    base_chunk = times // NUM_WORKERS
    chunks = [base_chunk] * NUM_WORKERS
    chunks[-1] += times - base_chunk * NUM_WORKERS   # remainder goes to last worker

    mp.set_start_method("spawn", force=True)
    args_list = [
        (c, N, image_size, interp_steps, slerp_dot_thr)
        for c in chunks
    ]

    with mp.Pool(NUM_WORKERS) as pool:
        results = pool.map(worker_fn, args_list)

    total_hits  = sum(r[0] for r in results)
    total_pairs = sum(r[1] for r in results)

    hit_rate = total_hits / total_pairs

    end = time.time()
    print(f"\n=== Results ===")
    print(f"Total pairs evaluated : {total_pairs:,}")
    print(f"Parallel-branch hits  : {total_hits:,}")
    print(f"Hit rate              : {hit_rate:.6f}  ({hit_rate*100:.4f}%)")
    print(f"Time: {end - start}")


'''
=== Results ===
Total pairs evaluated : 1,000,000
Parallel-branch hits  : 0
Hit rate              : 0.000000  (0.0000%)
Time: 77.94987869262695 
'''