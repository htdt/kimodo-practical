"""Convert a shipped Kimodo demo-example motion.npz into kimogen's
canonicalized NPZ format, so bake_kimodo.py + the browser viewers can be
tested before the model even runs locally.

Usage: python convert_example.py <example_dir_or_npz> <name> [--out out/examples]
"""
import argparse
import json
import os

import numpy as np

from kimogen import canonicalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("name")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out/examples"))
    a = ap.parse_args()
    src = a.src if a.src.endswith(".npz") else os.path.join(a.src, "motion.npz")
    os.makedirs(a.out, exist_ok=True)

    z = np.load(src)
    pj = z["posed_joints"]
    J = pj.shape[1]
    from kimodo.skeleton.definitions import SOMASkeleton30, SOMASkeleton77
    sk = SOMASkeleton77() if J == 77 else SOMASkeleton30()
    idx = {n: i for i, n in enumerate(sk.bone_order_names)}

    rp = z["root_positions"] if "root_positions" in z else pj[:, sk.root_idx]
    jc, rc, R = canonicalize(pj, rp, idx)
    grm = np.einsum("ij,tnjk->tnik", R, z["global_rot_mats"])
    fc = z["foot_contacts"] if "foot_contacts" in z else np.zeros((len(pj), 4))

    np.savez_compressed(
        os.path.join(a.out, f"{a.name}.npz"),
        posed_joints=jc.astype(np.float32),
        root_positions=rc.astype(np.float32),
        global_rot_mats=grm.astype(np.float32),
        foot_contacts=fc.astype(np.float32),
        fps=30)
    print(f"[example] {a.name}: {len(pj)} frames, {J} joints -> {a.out}/{a.name}.npz")


if __name__ == "__main__":
    main()
