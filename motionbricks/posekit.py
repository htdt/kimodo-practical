"""posekit — mine keyframe poses from G1 qpos CSVs (LAFAN1 retargeting) for the
keyframe-conditioned animation pipeline (BAKE.md §2).

A "pose" is a G1 mujoco qpos row (36,) = root pos(3) + root quat WXYZ(4) + 29 dof.
The library stores 4-frame windows (the model's target keyframe unit): a window
centered on a dynamic pose keeps its momentum; a single frame tiled 4x means
"arrive and hold".

Commands (run from GR00T-WholeBodyControl/motionbricks/, no GPU needed):
  scan <csv>                    print top candidate frames per heuristic tag
  sheet <csv> --frames 100,200  render a stick-figure contact sheet PNG
  sheet <csv> --range a:b:step  ... for a frame range
  save <name> <csv> <frame> [--window 4] [--hold]   add pose to the library
  show <name> [...]             contact sheet of saved library poses
  list                          list library poses

Library: pose_library.json (qpos embedded, self-contained). A starter library
ships with this repo — mining CSVs is only needed to ADD poses. CSVs are G1
retargeted LAFAN1 mocap (Unitree convention: root pos + XYZW quat + 29 dof),
e.g. the `g1/` folder of huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset;
point POSEKIT_CSV_DIR at the folder. The library stores WXYZ (mujoco).
"""
import argparse
import json
import os
import sys

import numpy as np
import mujoco

os.chdir(os.path.dirname(os.path.abspath(__file__)))

XML = "assets/skeletons/g1/scene_29dof.xml"
LIB = "pose_library.json"
CSV_DIR = os.environ.get("POSEKIT_CSV_DIR", "lafan_g1")

BODY = {}  # name -> body id, filled in load_model

# wrist dof columns in qpos: [roll, pitch, yaw] per side
WRIST_L, WRIST_R = [26, 27, 28], [33, 34, 35]
# Wrist dofs are AUTHORED control channels, never trusted from data: the
# LAFAN1->G1 IK leaves them arbitrary (they barely affect its objective), so
# mined keyframes carry meaningless twists that read as broken hands after
# retargeting. save() overwrites them with this neutral; `posekit.py wrists`
# sets deliberate per-pose values. The neutral (65deg inward roll = vertical
# palm facing the opponent) was picked empirically on the target rig with a
# roll/pitch/yaw sweep — 0 roll leaves the palm flat/down ("jazz hand").
WRIST_NEUTRAL_L, WRIST_NEUTRAL_R = [1.13, 0.0, 0.0], [-1.13, 0.0, 0.0]


def neutralize_wrists(qpos):
    """qpos (..., 36): overwrite wrist dofs with the neutral pose."""
    q = np.asarray(qpos, dtype=float).copy()
    q[..., WRIST_L] = WRIST_NEUTRAL_L
    q[..., WRIST_R] = WRIST_NEUTRAL_R
    return q


def load_model():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    for i in range(m.nbody):
        BODY[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)] = i
    return m, d


def load_csv(path):
    """CSV rows -> mujoco qpos (WXYZ root quat)."""
    if not os.path.exists(path) and not os.path.isabs(path):
        path = os.path.join(CSV_DIR, path)
    raw = np.loadtxt(path, delimiter=",")
    q = raw.copy()
    q[:, 3] = raw[:, 6]
    q[:, 4:7] = raw[:, 3:6]
    return q


def fk(m, d, qpos):
    """FK all frames -> xpos (F, nbody, 3), xquat (F, nbody, 4 wxyz)."""
    F = len(qpos)
    xpos = np.zeros((F, m.nbody, 3))
    xquat = np.zeros((F, m.nbody, 4))
    for i in range(F):
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(m, d)
        xpos[i] = d.xpos
        xquat[i] = d.xquat
    return xpos, xquat


def heading_frame(xquat_pelvis):
    """Yaw angle of the pelvis from its wxyz quat (Z-up)."""
    w, x, y, z = xquat_pelvis.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def features(m, xpos, xquat):
    """Per-frame feature table used by the tag heuristics (Z-up mujoco space)."""
    B = BODY
    pel = xpos[:, B["pelvis"]]
    yaw = heading_frame(xquat[:, B["pelvis"]])
    fwd = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)          # heading dir (x,y)
    la, ra = xpos[:, B["left_ankle_roll_link"]], xpos[:, B["right_ankle_roll_link"]]
    lw, rw = xpos[:, B["left_wrist_yaw_link"]], xpos[:, B["right_wrist_yaw_link"]]
    torso = xpos[:, B["torso_link"]]

    def reach(p):  # forward reach of a point in the heading frame, pelvis-relative
        rel = p[:, :2] - pel[:, :2]
        return (rel * fwd).sum(axis=1)

    def speed(p):
        v = np.zeros(len(p))
        v[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) * 30.0
        return v

    # torso lean: angle of pelvis->torso vs vertical, signed by heading component
    up = torso - pel
    lean = np.degrees(np.arctan2((up[:, :2] * fwd).sum(axis=1), up[:, 2]))

    return dict(
        pelvis_z=pel[:, 2], yaw=yaw,
        l_ankle_z=la[:, 2], r_ankle_z=ra[:, 2],
        l_ankle_speed=speed(la), r_ankle_speed=speed(ra),
        l_wrist_z=lw[:, 2], r_wrist_z=rw[:, 2],
        l_wrist_reach=reach(lw), r_wrist_reach=reach(rw),
        l_wrist_speed=speed(lw), r_wrist_speed=speed(rw),
        l_ankle_reach=reach(la), r_ankle_reach=reach(ra),
        torso_lean=lean, root_speed=speed(pel),
    )


def local_maxima(x, order=8):
    idx = []
    for i in range(order, len(x) - order):
        w = x[i - order:i + order + 1]
        if x[i] == w.max() and x[i] > w.min():
            idx.append(i)
    return np.array(idx, dtype=int)


def candidates(f):
    """tag -> list of (frame, score) candidate keyframes."""
    out = {}

    def top(idx, score, n=12):
        if len(idx) == 0:
            return []
        s = score[idx]
        order = np.argsort(-s)
        return [(int(idx[i]), float(s[i])) for i in order[:n]]

    ankle_hi = np.maximum(f["l_ankle_z"], f["r_ankle_z"])
    both_air = np.minimum(f["l_ankle_z"], f["r_ankle_z"])
    wrist_reach = np.maximum(f["l_wrist_reach"], f["r_wrist_reach"])
    wrist_speed = np.maximum(f["l_wrist_speed"], f["r_wrist_speed"])
    ankle_reach = np.maximum(f["l_ankle_reach"], f["r_ankle_reach"])

    idx = local_maxima(ankle_hi, 10)
    out["kick_apex"] = top(idx[ankle_hi[idx] > 0.45], ankle_hi)

    # punch extension: forward wrist reach peak while the hand is moving fast
    idx = local_maxima(wrist_reach, 6)
    m = (wrist_reach[idx] > 0.30) & (wrist_speed[idx] > 1.0)
    out["punch_ext"] = top(idx[m], wrist_reach + 0.1 * wrist_speed)

    idx = local_maxima(-f["pelvis_z"], 10)
    out["crouch"] = top(idx[f["pelvis_z"][idx] < 0.55], -f["pelvis_z"])

    idx = local_maxima(both_air, 6)
    out["jump_air"] = top(idx[both_air[idx] > 0.12], both_air + f["pelvis_z"])

    # guard: both wrists high (chest/chin), slow, upright
    guard = (f["l_wrist_z"] > 0.95) & (f["r_wrist_z"] > 0.95) & \
            (f["l_wrist_speed"] < 0.6) & (f["r_wrist_speed"] < 0.6) & \
            (np.abs(f["torso_lean"]) < 20) & (f["pelvis_z"] > 0.6)
    out["guard"] = top(np.where(guard)[0][::10], f["l_wrist_z"] + f["r_wrist_z"])

    # sweep / low kick: fast ankle, low, reaching forward, pelvis lowered
    fast_low = np.minimum(f["l_ankle_speed"] + f["r_ankle_speed"], 10)
    idx = local_maxima(fast_low, 8)
    m = (ankle_hi[idx] < 0.35) & (ankle_reach[idx] > 0.25) & (f["pelvis_z"][idx] < 0.68)
    out["sweep_low"] = top(idx[m], fast_low)

    # lean back (hit reaction / dodge)
    idx = local_maxima(-f["torso_lean"], 8)
    out["lean_back"] = top(idx[f["torso_lean"][idx] < -18], -f["torso_lean"])

    # high stance reach up (victory-ish): both wrists overhead
    over = (f["l_wrist_z"] > 1.25) & (f["r_wrist_z"] > 1.25)
    out["arms_up"] = top(np.where(over)[0][::8], f["l_wrist_z"] + f["r_wrist_z"])
    return out


# ---------- rendering ----------

BONES = [  # (parent, child) body-name pairs for the stick figure
    ("pelvis", "left_hip_roll_link"), ("left_hip_roll_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_roll_link"), ("right_hip_roll_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("pelvis", "torso_link"),
    ("torso_link", "left_shoulder_roll_link"), ("left_shoulder_roll_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_roll_link"), ("right_shoulder_roll_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
]


def draw_pose(ax, xpos, yaw, view="front"):
    """Project the skeleton into the heading frame: front = seen from +forward."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s], [s, c]])
    pts = {}
    for n, i in BODY.items():
        p = xpos[i]
        xy = R @ (p[:2] - xpos[BODY["pelvis"]][:2])
        pts[n] = (xy[0], xy[1], p[2])  # (fwd, left, up)
    for a, b in BONES:
        pa, pb = pts[a], pts[b]
        if view == "front":   # looking at the character from the front: x=left(mirrored), y=up
            ax.plot([-pa[1], -pb[1]], [pa[2], pb[2]], "o-", lw=2, ms=3,
                    color="#d33" if "right" in a or "right" in b else "#36c")
        else:                 # side view: x=fwd, y=up
            ax.plot([pa[0], pb[0]], [pa[2], pb[2]], "o-", lw=2, ms=3,
                    color="#d33" if "right" in a or "right" in b else "#36c")
    ax.axhline(0, color="#999", lw=0.5)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.15, 1.9)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def sheet(m, d, qpos, frames, out_png, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xpos, xquat = fk(m, d, qpos[frames])
    yaws = heading_frame(xquat[:, BODY["pelvis"]])
    n = len(frames)
    cols = min(n, 6)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows * 2, cols, figsize=(2.0 * cols, 4.6 * rows))
    axes = np.atleast_2d(axes)
    for k in range(rows * cols):
        r, c = divmod(k, cols)
        af, as_ = axes[2 * r, c], axes[2 * r + 1, c]
        if k < n:
            draw_pose(af, xpos[k], yaws[k], "front")
            draw_pose(as_, xpos[k], yaws[k], "side")
            af.set_title(f"f{frames[k]}", fontsize=9)
        else:
            af.axis("off"); as_.axis("off")
    fig.suptitle(f"{title}  (top: front, bottom: side)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print(f"[sheet] wrote {out_png} ({n} poses)")


# ---------- library ----------

def lib_load():
    if os.path.exists(LIB):
        with open(LIB) as fp:
            return json.load(fp)
    return {}


def lib_save(lib):
    if os.path.dirname(LIB):
        os.makedirs(os.path.dirname(LIB), exist_ok=True)
    with open(LIB, "w") as fp:
        json.dump(lib, fp)
    print(f"[lib] saved {LIB} ({len(lib)} poses)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan"); s.add_argument("csv")
    s = sub.add_parser("sheet"); s.add_argument("csv")
    s.add_argument("--frames", type=str, default=None)
    s.add_argument("--range", type=str, default=None, dest="rng")
    s.add_argument("--out", type=str, default="sheet.png")
    s = sub.add_parser("save")
    s.add_argument("name"); s.add_argument("csv"); s.add_argument("frame", type=int)
    s.add_argument("--window", type=int, default=4,
                   help="frames in the target window (>=1); centered on `frame`")
    s.add_argument("--hold", action="store_true",
                   help="tile the single frame instead of a moving window (arrive-and-hold)")
    s.add_argument("--keep-wrists", action="store_true",
                   help="keep the source wrist dofs (default: neutralize them)")
    s = sub.add_parser("show"); s.add_argument("names", nargs="+")
    s.add_argument("--out", type=str, default="lib_sheet.png")
    sub.add_parser("list")
    s = sub.add_parser("wrists",
                       help="author wrist dofs on saved poses (degrees)")
    s.add_argument("names", nargs="+", help="library pose names (or 'all')")
    s.add_argument("--left", type=str, default=None,
                   help="roll,pitch,yaw deg (default: neutral)")
    s.add_argument("--right", type=str, default=None,
                   help="roll,pitch,yaw deg (default: neutral, i.e. mirrored)")
    a = p.parse_args()

    m, d = load_model()

    if a.cmd == "scan":
        q = load_csv(a.csv)
        xpos, xquat = fk(m, d, q)
        f = features(m, xpos, xquat)
        for tag, cands in candidates(f).items():
            print(f"\n[{tag}]")
            for fr, sc in cands:
                print(f"  f{fr:5d} score {sc:6.3f}  pelvis {f['pelvis_z'][fr]:.2f} "
                      f"ankle {max(f['l_ankle_z'][fr], f['r_ankle_z'][fr]):.2f} "
                      f"reach {max(f['l_wrist_reach'][fr], f['r_wrist_reach'][fr]):.2f}")

    elif a.cmd == "sheet":
        q = load_csv(a.csv)
        if a.frames:
            frames = [int(x) for x in a.frames.split(",")]
        elif a.rng:
            parts = [int(x) for x in a.rng.split(":")]
            frames = list(range(*parts))
        else:
            sys.exit("--frames or --range required")
        sheet(m, d, q, frames, a.out, title=os.path.basename(a.csv))

    elif a.cmd == "save":
        q = load_csv(a.csv)
        lib = lib_load()
        if a.hold or a.window == 1:
            w = np.tile(q[a.frame][None], (4, 1))
        else:
            half = a.window // 2
            i0 = max(0, a.frame - half)
            w = q[i0:i0 + a.window]
            if len(w) < 4:  # pad by repeating the last frame
                w = np.concatenate([w, np.tile(w[-1:], (4 - len(w), 1))])
            elif a.window > 4:  # resample to the 4-frame unit
                sel = np.linspace(0, len(w) - 1, 4).round().astype(int)
                w = w[sel]
        if not a.keep_wrists:
            w = neutralize_wrists(w)
        lib[a.name] = dict(csv=os.path.basename(a.csv), frame=a.frame,
                           hold=bool(a.hold), qpos=np.round(w, 5).tolist())
        lib_save(lib)

    elif a.cmd == "show":
        lib = lib_load()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(a.names)
        fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.6))
        axes = np.atleast_2d(axes.reshape(2, n))
        for k, name in enumerate(a.names):
            q = np.array(lib[name]["qpos"])[-1]  # arrival frame of the window
            xpos, xquat = fk(m, d, q[None])
            yaw = heading_frame(xquat[:, BODY["pelvis"]])[0]
            draw_pose(axes[0, k], xpos[0], yaw, "front")
            draw_pose(axes[1, k], xpos[0], yaw, "side")
            axes[0, k].set_title(name, fontsize=9)
        fig.tight_layout(); fig.savefig(a.out, dpi=110)
        print(f"[show] wrote {a.out}")

    elif a.cmd == "list":
        lib = lib_load()
        for k, v in lib.items():
            wl = np.degrees(np.array(v["qpos"])[-1, WRIST_L])
            print(f"{k:24s} {v['csv']} f{v['frame']} hold={v.get('hold', False)} "
                  f"wristL[{wl[0]:.0f},{wl[1]:.0f},{wl[2]:.0f}]deg")

    elif a.cmd == "wrists":
        lib = lib_load()
        left = np.radians([float(x) for x in a.left.split(",")]) if a.left \
            else np.array(WRIST_NEUTRAL_L)
        right = np.radians([float(x) for x in a.right.split(",")]) if a.right \
            else np.array(WRIST_NEUTRAL_R)
        names = list(lib) if a.names == ["all"] else a.names
        for name in names:
            q = np.array(lib[name]["qpos"])
            q[:, WRIST_L] = left
            q[:, WRIST_R] = right
            lib[name]["qpos"] = np.round(q, 5).tolist()
            print(f"[wrists] {name}: L {np.degrees(left).round(1)} "
                  f"R {np.degrees(right).round(1)} deg")
        lib_save(lib)


if __name__ == "__main__":
    main()
