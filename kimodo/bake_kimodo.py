"""bake_kimodo — export Kimodo move NPZs as browser motion JSONs + manifest.

Kimodo counterpart of motionbricks/bake_moves.py. Clips are already
canonicalized by kimogen (frame-0 root at origin facing +X) and already Y-up
(Kimodo == three.js axes, no basis change — the G1 pipeline's mujoco Z-up
conversion is gone). Each JSON carries `srcMap` (SOMA-77 joint names →
canonical source roles) so align/retarget.js drives any certified rig from it
without knowing the source family.

Usage: kimenv/bin/python bake_kimodo.py [--in out/moves] [--web ../web/moves_kimodo]
No GPU needed.
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))

ORDER = ["idle_stance", "walk_fwd", "walk_back", "jump_up", "crouch",
         "block_high", "jab", "punch_heavy", "uppercut", "kick_front",
         "kick_high", "kick_side", "sweep", "hit_head", "hit_heavy",
         "knockdown", "victory"]

# canonical source roles -> somaskel77 joint names (mirror of SOMA_SRC in
# align/retarget.js; baked into each clip JSON as data.srcMap)
SOMA_SRC = {
    "Hips": "Hips", "Chest": "Chest",
    "LeftHipAnchor": "LeftLeg", "RightHipAnchor": "RightLeg",
    "LeftShoulderAnchor": "LeftArm", "RightShoulderAnchor": "RightArm",
    "LeftUpLeg": "LeftLeg", "LeftLeg": "LeftShin", "LeftFoot": "LeftFoot",
    "RightUpLeg": "RightLeg", "RightLeg": "RightShin", "RightFoot": "RightFoot",
    "LeftArm": "LeftArm", "LeftForeArm": "LeftForeArm", "LeftHand": "LeftHand",
    "RightArm": "RightArm", "RightForeArm": "RightForeArm", "RightHand": "RightHand",
}


def mats_to_xyzw(m):
    """[...,3,3] rotation matrices -> [...,4] xyzw quats (three.js order)."""
    shape = m.shape[:-2]
    q = Rot.from_matrix(m.reshape(-1, 3, 3)).as_quat()   # xyzw
    return q.reshape(*shape, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=os.path.join(HERE, "out/moves"))
    ap.add_argument("--web-dir", default=os.path.join(HERE, "../web/moves_kimodo"))
    ap.add_argument("--all", action="store_true",
                    help="bake every npz in in-dir instead of the MK ORDER list")
    a = ap.parse_args()
    os.makedirs(a.web_dir, exist_ok=True)

    from kimodo.skeleton.definitions import SOMASkeleton30, SOMASkeleton77

    def skel_pack(J):
        """names/parents/rest/restQuat for a 77- or 30-joint SOMA export."""
        sk77 = SOMASkeleton77()
        if J == 77:
            sk, rots = sk77, sk77.global_rot_offsets.numpy()
        else:
            sk = SOMASkeleton30()
            rots = sk77.global_rot_offsets.numpy()[sk.get_skel_slice(sk77)]
        names = sk.bone_order_names
        parents = [-1 if p is None else names.index(p)
                   for _, p in sk.bone_order_names_with_parents]
        # rest pose: standard T-pose, lifted so the lowest foot joint sits on
        # the ground (the retargeter reads rest hip/ankle heights from this),
        # and yawed -90° so it faces +X like every canonicalized clip (the
        # SOMA T-pose faces +Z; the retargeter yaw-rebases from the source
        # REST heading, so rest and clips must agree)
        c, s = 0.0, 1.0                                   # rot_y(pi/2): +Z -> +X
        Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        rest = sk.neutral_joints.numpy() @ Ry.T
        foot_i = [names.index(n) for n in
                  ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]]
        rest[:, 1] -= rest[foot_i, 1].min()
        rest_quat = mats_to_xyzw(np.einsum("ij,njk->nik", Ry, rots))
        return names, parents, rest, rest_quat

    packs = {}

    order = ORDER if not a.all else sorted(
        f[:-4] for f in os.listdir(a.in_dir) if f.endswith(".npz"))
    manifest = []
    for name in order:
        src = os.path.join(a.in_dir, f"{name}.npz")
        if not os.path.exists(src):
            print(f"[skip] {name} (no npz)")
            continue
        z = np.load(src)
        pj, grm = z["posed_joints"], z["global_rot_mats"]
        F, J = pj.shape[0], pj.shape[1]
        if J not in packs:
            packs[J] = skel_pack(J)
        names, parents, rest, rest_quat = packs[J]
        quat = mats_to_xyzw(grm)

        out = {
            "fps": int(z["fps"]), "numFrames": F, "mode": name,
            "names": names, "parents": parents,
            "pos": np.round(pj, 4).tolist(),
            "rest": np.round(rest, 4).tolist(),
            "quat": np.round(quat, 5).tolist(),
            "restQuat": np.round(rest_quat, 5).tolist(),
            "srcMap": SOMA_SRC,
            "source": "kimodo-soma-rp-v1.1",
        }
        with open(os.path.join(a.web_dir, f"{name}.json"), "w") as fp:
            json.dump(out, fp)

        meta_path = os.path.join(a.in_dir, f"{name}.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        manifest.append({"name": name, "file": f"{name}.json", "frames": F,
                         "frame_data": meta.get("frame_data"),
                         "gates": {k: v for k, v in meta.get("gates", {}).items()
                                   if not isinstance(v, list)}})
        print(f"[bake] {name}: {F} frames")

    with open(os.path.join(a.web_dir, "manifest.json"), "w") as fp:
        json.dump({"moves": manifest, "source": "kimodo"}, fp, indent=1)
    print(f"[bake] manifest: {len(manifest)} moves -> {a.web_dir}/manifest.json")


if __name__ == "__main__":
    main()
