"""
Coordinate conversion MuJoCo (Z-up) -> three.js (Y-up) and the canonical
G1-link -> humanoid-bone map. bake_moves.py imports MAP / to_three_pos /
to_three_quat from here.

Standalone use: convert a single npz (G1 world transforms from MuJoCo FK)
into a delta-retargeting motion JSON — per mapped humanoid bone, the
WORLD-space delta rotation of the matching G1 link relative to a neutral rest
pose, plus the pelvis world translation. A runtime composes these deltas onto
the GLB's bind pose (world-space delta retargeting), which is robust to
robot<->human proportion differences.
"""
import json
import sys
import numpy as np

# Mixamo bone (in character.glb)  <-  G1 link (MuJoCo body)
# NOTE: this rig's spine chain is Hips -> Spine02 -> Spine01 -> Spine(chest),
# so Spine02 is the LOWEST spine and Spine is the chest that carries the arms.
# Thigh/upper-arm use the hip_ROLL / shoulder_ROLL links (anatomical joint centers);
# the yaw links sit along the limb with a forward offset that skews the segment.
MAP = {
    "Hips":         "pelvis",
    "Spine02":      "waist_yaw_link",
    "Spine01":      "waist_roll_link",
    "Spine":        "torso_link",
    "LeftUpLeg":    "left_hip_roll_link",
    "LeftLeg":      "left_knee_link",
    "LeftFoot":     "left_ankle_roll_link",
    "RightUpLeg":   "right_hip_roll_link",
    "RightLeg":     "right_knee_link",
    "RightFoot":    "right_ankle_roll_link",
    "LeftArm":      "left_shoulder_roll_link",
    "LeftForeArm":  "left_elbow_link",
    "LeftHand":     "left_wrist_yaw_link",
    "RightArm":     "right_shoulder_roll_link",
    "RightForeArm": "right_elbow_link",
    "RightHand":    "right_wrist_yaw_link",
}


def qmul(a, b):  # xyzw, broadcasting
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def qconj(a):
    return np.stack([-a[..., 0], -a[..., 1], -a[..., 2], a[..., 3]], axis=-1)


def qinv(a):
    return qconj(a) / np.sum(a * a, axis=-1, keepdims=True)


def qnorm(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


# basis change MuJoCo (Z-up, right-handed) -> three.js (Y-up): rotate -90deg about X
# (x, y, z)_mj -> (x, z, -y)_three   ;   as quaternion c (xyzw)
C = np.array([-0.7071067811865476, 0.0, 0.0, 0.7071067811865476])
CINV = qconj(C)


def wxyz_to_xyzw(q):
    return np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], axis=-1)


def to_three_quat(q_wxyz):
    q = wxyz_to_xyzw(q_wxyz)
    cc = np.broadcast_to(C, q.shape)
    ci = np.broadcast_to(CINV, q.shape)
    return qnorm(qmul(qmul(cc, q), ci))


def to_three_pos(p):  # (...,3): (x,y,z)_mj -> (x, z, -y)_three
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "motion_g1.npz"
    dst = sys.argv[2] if len(sys.argv) > 2 else "motion.json"
    d = np.load(src, allow_pickle=True)
    xquat, xpos = d["xquat"], d["xpos"]            # (F,nbody,4 wxyz), (F,nbody,3)
    rest_xquat, rest_xpos = d["rest_xquat"], d["rest_xpos"]
    names = [str(x) for x in d["body_names"]]
    fps = int(d["fps"])
    idx = {n: i for i, n in enumerate(names)}
    F = xquat.shape[0]

    q3 = to_three_quat(xquat)                       # (F,nbody,4) three-space world quats
    qrest3 = to_three_quat(rest_xquat)              # (nbody,4)
    p3 = to_three_pos(xpos)
    prest3 = to_three_pos(rest_xpos)

    missing = [b for b in MAP.values() if b not in idx]
    if missing:
        raise SystemExit(f"bodies not found in npz: {missing}\navailable={names}")

    delta = {}
    for bone, body in MAP.items():
        bi = idx[body]
        rest_inv = qinv(qrest3[bi])[None, :]                         # (1,4)
        dq = qmul(q3[:, bi, :], np.broadcast_to(rest_inv, (F, 4)))   # world delta (F,4)
        delta[bone] = np.round(qnorm(dq), 6).tolist()

    pelvis = idx["pelvis"]
    root_pos = p3[:, pelvis, :] - prest3[pelvis][None, :]            # rel. to rest, three-space
    g1_hip_height = float(prest3[pelvis][1])                          # y in three-space (~0.8)

    out = {
        "fps": fps,
        "numFrames": F,
        "g1HipHeight": g1_hip_height,
        "mode": str(d["mode"]) if "mode" in d else "walk",
        "rootPos": np.round(root_pos, 5).tolist(),
        "delta": delta,
        "mapped": list(MAP.keys()),
    }
    with open(dst, "w") as f:
        json.dump(out, f)
    print(f"[json] wrote {dst}  frames={F} fps={fps} g1HipHeight={g1_hip_height:.3f}")
    print(f"[json] mapped bones: {list(MAP.keys())}")


if __name__ == "__main__":
    main()
