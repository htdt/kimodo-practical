"""bake_moves — normalize generated move npz clips (movegen.py output) and
export per-move motion JSONs + a manifest, ready for the ALIGN-stage runtime
(retarget.js / certify.mjs) or any engine importer (BAKE.md §5).

Normalization: rotate/translate every clip so frame 0 has the pelvis at the
origin with heading 0 (mujoco +X). Guarantees all baked moves share one
canonical start, whatever world frame their stance keyframe came from.

Usage (from GR00T-WholeBodyControl/motionbricks/, no GPU needed — pure
mujoco FK):
  python bake_moves.py [--in-dir out/moves] [--out-dir baked]
                       [--spec moves_example.json]
Moves are exported in spec order when --spec is readable, else every npz in
--in-dir sorted by name.
"""
import argparse
import glob
import json
import os

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from make_motion_json import to_three_pos, to_three_quat, MAP

XML = "assets/skeletons/g1/scene_29dof.xml"


def move_order(in_dir, spec_path):
    if spec_path and os.path.exists(spec_path):
        with open(spec_path) as fp:
            return [m["name"] for m in json.load(fp)["moves"]]
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(in_dir, "*.npz")))


def canonicalize(qpos):
    """Frame-0 pelvis -> origin, heading -> 0 (yaw about Z). qpos (F, 36) WXYZ."""
    q = qpos.copy()
    rot0 = R.from_quat(q[0, 3:7][[1, 2, 3, 0]])  # wxyz -> xyzw
    yaw0 = rot0.as_euler("zyx")[0]
    unyaw = R.from_euler("z", -yaw0)
    p0 = q[0, :3] * [1.0, 1.0, 0.0]
    q[:, :3] = unyaw.apply(q[:, :3] - p0)
    rots = R.from_quat(q[:, 3:7][:, [1, 2, 3, 0]])
    out = (unyaw * rots).as_quat()          # xyzw
    q[:, 3:7] = out[:, [3, 0, 1, 2]]        # -> wxyz
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="out/moves")
    ap.add_argument("--out-dir", default="baked")
    ap.add_argument("--spec", default="moves_example.json")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    nbody = m.nbody
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(nbody)]
    parents = [int(m.body_parentid[i]) for i in range(nbody)]

    mujoco.mj_resetData(m, d)
    d.qpos[:] = 0.0
    d.qpos[0:3] = [0.0, 0.0, 0.8]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    rest_xpos, rest_xquat = d.xpos.copy(), d.xquat.copy()

    manifest = []
    for name in move_order(a.in_dir, a.spec):
        src = os.path.join(a.in_dir, f"{name}.npz")
        if not os.path.exists(src):
            print(f"[skip] {name} (no npz)")
            continue
        z = np.load(src, allow_pickle=True)
        q = canonicalize(z["qpos"])
        F = len(q)
        xpos = np.zeros((F, nbody, 3), dtype=np.float32)
        xquat = np.zeros((F, nbody, 4), dtype=np.float32)
        for i in range(F):
            d.qpos[:] = q[i]
            mujoco.mj_forward(m, d)
            xpos[i] = d.xpos
            xquat[i] = d.xquat

        p3, q3 = to_three_pos(xpos), to_three_quat(xquat)
        out = {
            "fps": int(z["fps"]), "numFrames": F, "mode": name,
            "names": names, "parents": parents,
            "pos": np.round(p3, 4).tolist(),
            "rest": np.round(to_three_pos(rest_xpos), 4).tolist(),
            "quat": np.round(q3, 5).tolist(),
            "restQuat": np.round(to_three_quat(rest_xquat), 5).tolist(),
            "mapSource": list(MAP.values()), "mapTarget": list(MAP.keys()),
        }
        with open(os.path.join(a.out_dir, f"{name}.json"), "w") as fp:
            json.dump(out, fp)

        meta_path = os.path.join(a.in_dir, f"{name}.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        manifest.append({"name": name, "file": f"{name}.json", "frames": F,
                         "frame_data": meta.get("frame_data"),
                         "gates": {k: round(v, 4) for k, v in
                                   meta.get("gates", {}).items()
                                   if isinstance(v, (int, float))}})
        print(f"[bake] {name}: {F} frames")

    with open(os.path.join(a.out_dir, "manifest.json"), "w") as fp:
        json.dump({"moves": manifest}, fp, indent=1)
    print(f"[bake] manifest: {len(manifest)} moves -> {a.out_dir}/manifest.json")


if __name__ == "__main__":
    main()
