"""Validate Kimodo axes on a generated forward-walk NPZ.

Checks the conventions consumed by kimogen.py: Y-up with feet near y=0,
hip-line forward aligned to travel, and heading stored as [cos(t), sin(t)]
with facing (sin(t), 0, cos(t)). Exits nonzero on a mismatch.
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="raw Kimodo forward-walk NPZ")
    args = ap.parse_args()

    z = np.load(args.npz)
    if "posed_joints" not in z:
        ap.error("NPZ has no posed_joints")
    if "global_root_heading" not in z:
        ap.error("raw Kimodo NPZ has no global_root_heading")
    joints = z["posed_joints"]
    if (joints.ndim != 3 or joints.shape[-1] != 3
            or joints.shape[0] < 2 or joints.shape[1] not in (30, 77)):
        ap.error(f"posed_joints must have shape [T,30|77,3], got {joints.shape}")
    if not np.isfinite(joints).all():
        print(json.dumps({"ok": False, "metrics": {},
                          "failures": ["motion contains non-finite values"]}, indent=2))
        raise SystemExit(1)

    from kimodo.skeleton.definitions import SOMASkeleton30, SOMASkeleton77
    skeleton = SOMASkeleton77() if joints.shape[1] == 77 else SOMASkeleton30()
    idx = {name: i for i, name in enumerate(skeleton.bone_order_names)}
    root = joints[:, skeleton.root_idx]

    right = joints[0, idx["RightLeg"]] - joints[0, idx["LeftLeg"]]
    forward = np.cross(np.array([0.0, 1.0, 0.0]), right)
    forward[1] = 0.0
    forward /= np.linalg.norm(forward) + 1e-12
    travel = root[-1] - root[0]
    travel[1] = 0.0
    travel_len = float(np.linalg.norm(travel))
    travel_dir = travel / (travel_len + 1e-12)

    toe_y = joints[:, [idx["LeftToeBase"], idx["RightToeBase"]], 1]
    metrics = {
        "frames": int(len(joints)),
        "joints": int(joints.shape[1]),
        "rootY": [round(float(root[:, 1].min()), 4), round(float(root[:, 1].max()), 4)],
        "toeMinY": round(float(toe_y.min()), 4),
        "travel": [round(float(x), 4) for x in travel],
        "hipForwardDotTravel": round(float(forward @ travel_dir), 4),
    }
    failures = []
    if abs(metrics["toeMinY"]) > 0.08:
        failures.append(f"ground is not near y=0 (toe min {metrics['toeMinY']} m)")
    if travel_len < 0.1:
        failures.append(f"forward-walk travel is too small ({travel_len:.3f} m)")
    if metrics["hipForwardDotTravel"] < 0.8:
        failures.append("hip-line forward does not align with travel")

    heading = z["global_root_heading"]
    if heading.ndim != 2 or heading.shape != (len(joints), 2):
        failures.append(f"global_root_heading has invalid shape {heading.shape}")
    elif not np.isfinite(heading).all():
        failures.append("global_root_heading contains non-finite values")
    else:
        heading_norm = float(np.linalg.norm(heading[0]))
        metrics["headingNorm"] = round(heading_norm, 4)
        if not 0.95 <= heading_norm <= 1.05:
            failures.append(f"heading [cos(t), sin(t)] is not unit length ({heading_norm:.3f})")
        angle = np.arctan2(heading[0, 1], heading[0, 0])
        heading_dir = np.array([np.sin(angle), 0.0, np.cos(angle)])
        metrics["headingDotTravel"] = round(float(heading_dir @ travel_dir), 4)
        if metrics["headingDotTravel"] < 0.8:
            failures.append("[cos(t), sin(t)] heading does not align with travel")

    print(json.dumps({"ok": not failures, "metrics": metrics, "failures": failures}, indent=2))
    z.close()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
