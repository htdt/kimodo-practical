"""Convert a shipped Kimodo demo-example motion.npz into kimogen's
canonicalized NPZ format, so bake_kimodo.py + the browser viewers can be
tested before the model even runs locally.

Usage: python convert_example.py <example_dir_or_npz> <name> [--out out/examples]
"""
import argparse
import os

import numpy as np

try:
    from .kimogen import SAFE_MOVE_NAME, canonicalize
except ImportError:  # direct script execution: python kimodo/convert_example.py
    from kimogen import SAFE_MOVE_NAME, canonicalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("name")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out/examples"))
    a = ap.parse_args()
    if not SAFE_MOVE_NAME.fullmatch(a.name):
        ap.error("name must contain only letters, numbers, '_' and '-'")
    src = a.src if a.src.endswith(".npz") else os.path.join(a.src, "motion.npz")
    os.makedirs(a.out, exist_ok=True)

    z = np.load(src)
    required = {"posed_joints", "global_rot_mats"}
    missing = required - set(z.files)
    if missing:
        raise SystemExit("source NPZ missing: " + ", ".join(sorted(missing)))
    pj = z["posed_joints"]
    if pj.ndim != 3 or pj.shape[-1] != 3:
        raise SystemExit(f"posed_joints must have shape [T,J,3], got {pj.shape}")
    J = pj.shape[1]
    if J not in (30, 77):
        raise SystemExit(f"expected a 30- or 77-joint SOMA clip, got {J}")
    grm = z["global_rot_mats"]
    if grm.shape != (len(pj), J, 3, 3):
        raise SystemExit(f"global_rot_mats must have shape [{len(pj)},{J},3,3]")
    if len(pj) < 2 or not np.isfinite(pj).all() or not np.isfinite(grm).all():
        raise SystemExit("motion must have at least two frames and contain only finite values")
    from kimodo.skeleton.definitions import SOMASkeleton30, SOMASkeleton77
    sk = SOMASkeleton77() if J == 77 else SOMASkeleton30()
    idx = {n: i for i, n in enumerate(sk.bone_order_names)}

    rp = z["root_positions"] if "root_positions" in z else pj[:, sk.root_idx]
    if rp.shape != (len(pj), 3) or not np.isfinite(rp).all():
        raise SystemExit(f"root_positions must have shape [{len(pj)},3] and be finite")
    jc, rc, R = canonicalize(pj, rp, idx)
    grm = np.einsum("ij,tnjk->tnik", R, grm)
    extra = {}
    if "local_rot_mats" in z:
        lrm = z["local_rot_mats"].copy()
        if lrm.shape != (len(pj), J, 3, 3) or not np.isfinite(lrm).all():
            raise SystemExit(f"local_rot_mats must have shape [{len(pj)},{J},3,3] and be finite")
        lrm[:, sk.root_idx] = np.einsum("ij,tjk->tik", R, lrm[:, sk.root_idx])
        extra["local_rot_mats"] = lrm.astype(np.float32)
    fc = z["foot_contacts"] if "foot_contacts" in z else np.zeros((len(pj), 4))
    if fc.ndim != 2 or fc.shape[0] != len(pj) or fc.shape[1] not in (4, 6) or not np.isfinite(fc).all():
        raise SystemExit("foot_contacts must have shape [T,4|6] and be finite")
    fps_value = np.asarray(z["fps"] if "fps" in z else 30)
    if fps_value.size != 1:
        raise SystemExit("fps must be a scalar")
    fps_float = float(fps_value.reshape(-1)[0])
    if (not np.isfinite(fps_float) or fps_float <= 0
            or abs(fps_float - round(fps_float)) > 1e-9):
        raise SystemExit("fps must be a positive integer")
    fps = int(round(fps_float))
    z.close()

    np.savez_compressed(
        os.path.join(a.out, f"{a.name}.npz"),
        posed_joints=jc.astype(np.float32),
        root_positions=rc.astype(np.float32),
        global_rot_mats=grm.astype(np.float32),
        foot_contacts=fc.astype(np.float32),
        canonical_rotation=R.astype(np.float32),
        fps=fps,
        **extra)
    print(f"[example] {a.name}: {len(pj)} frames, {J} joints -> {a.out}/{a.name}.npz")


if __name__ == "__main__":
    main()
