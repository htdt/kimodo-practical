"""movegen — keyframe-conditioned move generation (BAKE.md §3–4).

Takes a move-set spec (JSON) whose moves are either
  * "keyframes": a schedule of pose-library keyframes (posekit.py) that the
    MotionBricks inbetweener is driven through, segment by segment, via the
    8-frame constraint window of motion_inference.predict() (4 context frames
    from the previous segment + 4 target frames from the keyframe window), or
  * "mode": a native clip-holder skill (walk, jump, ...) rolled out with
    directional intent — used for locomotion cycles where the model's own
    prior is already the right source.

Each move is generated with N stochastic seeds (gumbel pose-token sampling),
scored by numeric QA gates (keyframe-hit error, foot skate, jitter, root
drift, joint limits) and the best seed is kept. Output per move:
  out/moves/<name>.npz    generate_motion.py-compatible (xpos/xquat/qpos/...)
  out/moves/<name>.json   gates for every seed + frame data (startup/active/
                          recovery from the keyframe schedule)

Usage (from GR00T-WholeBodyControl/motionbricks/, in the MotionBricks env —
see MOTIONBRICKS.md for setup):
  xvfb-run -a python movegen.py --spec moves_example.json
  ... --only jab,kick_front --seeds 8 --out-dir out/moves
"""
import argparse
import json
import os
import sys

import numpy as np
import torch as t
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from mbstack import build_args, DEVICE
from motionbricks.motion_backbone.demo.utils import navigation_demo
from motionbricks.motionlib.core.utils.rotations import angle_to_Y_rotation_matrix

FPS = 30
NFPT = 4          # NUM_FRAMES_PER_TOKEN
MIN_TOKENS, MAX_TOKENS = 6, 16
POSE_LIB = "pose_library.json"

# config overrides injected into every motion_inference.predict call
GEN_CONFIG = {"pose_token_sampling_use_argmax": False}


# ---------------------------------------------------------------- setup

def build_stack():
    demo = navigation_demo(build_args())
    agent = demo.full_agent
    agent.eval()

    inf = agent._inferencer
    orig_predict = inf.predict

    def predict_patched(*a, config=None, **kw):
        cfg = dict(config or {})
        cfg.update(GEN_CONFIG)
        return orig_predict(*a, config=cfg, **kw)

    inf.predict = predict_patched
    return demo, agent


def load_pose_lib():
    with open(POSE_LIB) as fp:
        lib = json.load(fp)
    return {k: np.array(v["qpos"], dtype=np.float32) for k, v in lib.items()}


# ---------------------------------------------------------------- geometry helpers

def yup_heading(rot):
    """Heading angle from world rotation matrices [..., 3, 3] in the converter's
    Y-up space (same formula the clip holder uses: forward = R @ +Z, yaw)."""
    fx = rot[..., 0, 2]
    fz = rot[..., 2, 2]
    return t.atan2(fx, fz)


def qpos_to_transforms(agent, qpos):
    """(F, 36) numpy WXYZ-> converter transforms pos [1,F,J,3], rot [1,F,J,3,3]."""
    q = t.tensor(qpos, dtype=t.float32, device=DEVICE)[None]
    with t.no_grad():
        pos, rot = agent._converter.convert_mujoco_qpos_to_motion_transforms(q)
    return pos, rot


# ---------------------------------------------------------------- segment generation

def gen_segment(agent, ctx_qpos, target_window, tokens, anchor_xz, heading, seed):
    """One predict() call: 4 context frames -> keyframe target window.

    ctx_qpos       (4, 36) world-space mujoco qpos (WXYZ)
    target_window  (4, 36) pose-library keyframe window (its own local frame)
    tokens         chunk length in 4-frame tokens (6..16) or None (model picks)
    anchor_xz      (2,) desired Y-up-space XZ of the window's first frame,
                   in the CANONICAL frame of this segment (see caller)
    heading        desired heading (canonical frame) of the window's first frame
    seed           torch manual seed for the gumbel pose-token sampling

    Returns world-space qpos (F, 36) for the whole chunk (first 4 frames
    reproduce the context).
    """
    inp = {
        "context_mujoco_qpos": t.tensor(ctx_qpos, dtype=t.float32, device=DEVICE)[None],
        "mode": t.tensor([[0]], dtype=t.long, device=DEVICE),
        "movement_direction": t.zeros([1, 3], device=DEVICE),
        "facing_direction": t.tensor([[1.0, 0.0, 0.0]], device=DEVICE),
        "random_seed": t.tensor([seed], dtype=t.long, device=DEVICE),
    }
    # canonicalizes the context (frame 0 -> origin, heading 0) and remembers
    # first_frame_position / first_frame_heading_angle for uncanonicalization
    inp["context_global_joint_positions"], inp["context_global_joint_rotations"] = \
        agent._process_input_to_joint_transforms(inp)

    # --- build the target constraint from the keyframe window ---
    pos, rot = qpos_to_transforms(agent, target_window)          # [1,4,J,3/3x3]
    gh = yup_heading(rot[:, :, 0])                               # [1,4] window headings
    diff = t.tensor(float(heading), device=DEVICE) - gh[:, 0]    # rigid yaw to desired
    R = angle_to_Y_rotation_matrix(diff).float()                 # [1,3,3]

    root_xz = pos[:, :, 0, :].clone()
    root_xz[..., 1] = 0.0                                        # (x, 0, z)
    joints_rel = pos - root_xz[:, :, None, :]                    # root-XZ-relative
    joints_rel = t.matmul(R[:, None, None], joints_rel[..., None])[..., 0]
    rot_out = t.matmul(R[:, None, None], rot)

    rel_xz = root_xz - root_xz[:, :1]                            # window's own travel
    rel_xz = t.matmul(R[:, None], rel_xz[..., None])[..., 0]
    anchor = t.tensor([anchor_xz[0], 0.0, anchor_xz[1]], device=DEVICE, dtype=t.float32)
    root_out = rel_xz + anchor[None, None, :]

    inp["target_global_joint_positions"] = joints_rel
    inp["target_global_joint_rotations"] = rot_out
    inp["target_global_root_positions"] = root_out
    inp["target_root_headings"] = gh + diff[:, None]

    if tokens is not None:
        tokens = int(np.clip(tokens, MIN_TOKENS, MAX_TOKENS))
        allowed = t.zeros([1, MAX_TOKENS - MIN_TOKENS + 1], dtype=t.long, device=DEVICE)
        allowed[0, tokens - MIN_TOKENS] = 1
        inp["allowed_pred_num_tokens"] = allowed

    t.manual_seed(seed)
    with t.no_grad():
        _, qpos, num_pred_frames = agent._generate_inbetween_frames(inp)
    return qpos[0, :num_pred_frames.item()].detach().cpu().numpy()


def canonical_ctx_info(agent, ctx_qpos):
    """Where the context ends up in ITS OWN canonical frame (Y-up space):
    returns (end_xz (2,), end_heading, fwd unit (2,)) for target anchoring."""
    q = ctx_qpos.copy()
    # canonicalize by frame 0 exactly like the agent does (Z-up qpos space)
    inp = {"context_mujoco_qpos": t.tensor(q, dtype=t.float32, device=DEVICE)[None],
           "movement_direction": t.zeros([1, 3], device=DEVICE),
           "facing_direction": t.tensor([[1.0, 0.0, 0.0]], device=DEVICE)}
    agent._canonicalize_mujoco_qpos(inp)
    pos, rot = agent._converter.convert_mujoco_qpos_to_motion_transforms(
        inp["context_mujoco_qpos"])
    gh = yup_heading(rot[:, :, 0])[0]                            # [4]
    end_xz = pos[0, -1, 0, [0, 2]].detach().cpu().numpy()
    h = float(gh[-1].item())
    fwd = np.array([np.sin(h), np.cos(h)])
    return end_xz, h, fwd


# ---------------------------------------------------------------- mode rollout

def gen_mode_move(agent, mode_name, chunks, direction, seed, start_qpos=None):
    """Native-skill rollout (walk fwd/back, jump...) with fixed facing +X.
    direction: 'fwd' | 'back' | 'none' (movement relative to facing)."""
    from mbstack import GenStep, default_allowed, make_inputs, MODE_NAMES
    genstep = GenStep(agent).eval()
    mode_idx = MODE_NAMES.index(mode_name)
    face = [1.0, 0.0, 0.0]
    move = {"fwd": [1.0, 0.0, 0.0], "back": [-1.0, 0.0, 0.0],
            "none": [0.0, 0.0, 0.0]}[direction]

    if start_qpos is None:
        agent.reset()
        buf = agent.frames["mujoco_qpos"][0].detach().cpu().numpy()
        ctx = buf[:4]
    else:
        ctx = start_qpos

    frames = []
    rng = np.random.default_rng(seed)
    for c in range(chunks):
        t.manual_seed(seed * 1000 + c)
        inputs = make_inputs(t.tensor(ctx, device=DEVICE)[None], mode_idx,
                             move, face, int(rng.integers(0, 10000)))
        with t.no_grad():
            qpos, npf = genstep(**inputs)
        q = qpos[0, :npf.item() * 1].detach().cpu().numpy()
        n = q.shape[0]
        frames.append(q[NFPT:] if c > 0 else q)
        ctx = q[n - NFPT:]
    return np.concatenate(frames, axis=0)


# ---------------------------------------------------------------- wrist authoring

WRIST_COLS = [26, 27, 28, 33, 34, 35]        # L roll/pitch/yaw, R roll/pitch/yaw


def author_wrists(qpos, anchors):
    """Wrist dofs are CONTROL channels, not model channels. Nothing upstream
    ever supplies real wrist signal (the LAFAN1->G1 IK leaves them arbitrary,
    the inbetweener's wrist channels are unregularized noise — the three tiny
    wrist links barely register in its features), so the model output is
    discarded wholesale and the channels are rebuilt as smoothstep
    interpolation between the authored library-pose values at each keyframe
    arrival. Hand orientation is thereby governed in exactly one place: the
    pose library (posekit.py wrists).

    anchors: [(frame_idx, wrist6)] — must include frame 0."""
    q = qpos.copy()
    F = len(q)
    anchors = sorted((min(int(f), F - 1), np.asarray(w, dtype=float))
                     for f, w in anchors)
    out = np.empty((F, 6))
    out[: anchors[0][0] + 1] = anchors[0][1]
    for (fa, wa), (fb, wb) in zip(anchors, anchors[1:]):
        if fb <= fa:
            continue
        s = np.linspace(0.0, 1.0, fb - fa + 1)
        s = s * s * (3.0 - 2.0 * s)
        out[fa:fb + 1] = wa + s[:, None] * (wb - wa)
    out[anchors[-1][0]:] = anchors[-1][1]
    q[:, WRIST_COLS] = out
    return q


# ---------------------------------------------------------------- QA gates

class Gates:
    def __init__(self, demo):
        self.m = demo.mj_model
        self.d = demo.mj_data
        self.bid = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i): i
                    for i in range(self.m.nbody)}
        # dof ranges (skip the free joint)
        self.lo = self.m.jnt_range[1:, 0].copy()
        self.hi = self.m.jnt_range[1:, 1].copy()

    def fk(self, qpos):
        F = len(qpos)
        xpos = np.zeros((F, self.m.nbody, 3), dtype=np.float32)
        xquat = np.zeros((F, self.m.nbody, 4), dtype=np.float32)
        for i in range(F):
            self.d.qpos[:] = qpos[i]
            mujoco.mj_forward(self.m, self.d)
            xpos[i] = self.d.xpos
            xquat[i] = self.d.xquat
        return xpos, xquat

    def eval(self, qpos, keyframes):
        """keyframes: list of (frame_idx, arrival_qpos (36,)) checkpoints."""
        xpos, xquat = self.fk(qpos)
        r = {}

        # --- foot skate: horizontal ankle speed while in contact ---
        skates = []
        for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            p = xpos[:, self.bid[name]]
            contact = p[:, 2] < (p[:, 2].min() + 0.03)
            v = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) * FPS
            c = contact[:-1] & contact[1:]
            if c.any():
                skates.append(v[c])
        sk = np.concatenate(skates) if skates else np.zeros(1)
        r["foot_skate_mean"] = float(sk.mean())
        r["foot_skate_p95"] = float(np.percentile(sk, 95))

        # --- jitter: 2nd difference of dof angles ---
        dd = np.abs(np.diff(qpos[:, 7:], n=2, axis=0))
        r["jitter_mean"] = float(dd.mean())
        r["jitter_max"] = float(dd.max())

        # --- joint limits ---
        viol = (qpos[:, 7:] < self.lo - 0.02) | (qpos[:, 7:] > self.hi + 0.02)
        r["limit_violation_frac"] = float(viol.mean())

        # --- keyframe hit: end-effector positions in the root frame ---
        errs = []
        for fi, kq in keyframes:
            fi = min(fi, len(qpos) - 1)
            ee_gen = self._ee_local(xpos[fi], xquat[fi])
            kx, kqt = self.fk(kq[None])
            ee_tgt = self._ee_local(kx[0], kqt[0])
            errs.append(float(np.linalg.norm(ee_gen - ee_tgt, axis=1).mean()))
        r["keyframe_ee_err"] = float(np.mean(errs)) if errs else 0.0
        r["keyframe_ee_err_max"] = float(np.max(errs)) if errs else 0.0
        return r

    def _ee_local(self, xpos, xquat):
        """Wrists+ankles+head-ish (torso top) relative to pelvis, yaw-derotated."""
        pel = xpos[self.bid["pelvis"]]
        w, x, y, z = xquat[self.bid["pelvis"]]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = np.cos(-yaw), np.sin(-yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        pts = [xpos[self.bid[n]] - pel for n in
               ("left_wrist_yaw_link", "right_wrist_yaw_link",
                "left_ankle_roll_link", "right_ankle_roll_link", "torso_link")]
        return (R @ np.array(pts).T).T


def score(g):
    """Lower is better; weights put keyframe fidelity first, then contact."""
    return (3.0 * g["keyframe_ee_err"] + 1.5 * g["foot_skate_mean"] +
            20.0 * g["jitter_mean"] + 2.0 * g["limit_violation_frac"])


# ---------------------------------------------------------------- move assembly

def gen_keyframe_move(agent, lib, move, seed):
    """Chain the keyframe schedule; returns (qpos (F,36), keyframe checkpoints,
    segment boundaries)."""
    start_pose = lib[move.get("start", "stance")][-1]
    ctx = np.tile(start_pose[None], (4, 1))
    # ground the start pose: keep as-is (library poses are already grounded)
    out = [ctx.copy()]
    checkpoints = []   # (frame index in final clip, arrival qpos)
    bounds = [4]
    total = 4

    for step in move["steps"]:
        window = lib[step["pose"]]
        end_xz, h, fwd = canonical_ctx_info(agent, ctx)
        side = np.array([fwd[1], -fwd[0]])
        dxy = step.get("dxy", [0.0, 0.0])
        anchor = end_xz + fwd * dxy[0] + side * dxy[1]
        heading = h + np.radians(step.get("turn", 0.0))

        q = gen_segment(agent, ctx, window, step.get("tokens"), anchor, heading, seed)
        out.append(q[NFPT:])
        total += len(q) - NFPT
        checkpoints.append((total - 1, window[-1]))
        bounds.append(total)
        ctx = q[-NFPT:]
    return np.concatenate(out, axis=0), checkpoints, bounds


def loop_trim(qpos, min_len=36, max_len=120):
    """Find the (i, j) pair minimizing dof-space distance for cycle moves."""
    F = len(qpos)
    best, bi, bj = 1e9, 0, min(F - 1, min_len)
    for i in range(0, max(1, F - min_len), 2):
        for j in range(i + min_len, min(F, i + max_len), 2):
            d = np.abs(qpos[i, 7:] - qpos[j, 7:]).mean() + \
                3.0 * abs(qpos[i, 2] - qpos[j, 2])
            if d < best:
                best, bi, bj = d, i, j
    return qpos[bi:bj], best


# ---------------------------------------------------------------- npz export

def export_npz(demo, qpos, path, name):
    m, d = demo.mj_model, demo.mj_data
    nbody = m.nbody
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(nbody)]
    F = len(qpos)
    xpos = np.zeros((F, nbody, 3), dtype=np.float32)
    xquat = np.zeros((F, nbody, 4), dtype=np.float32)
    for i in range(F):
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(m, d)
        xpos[i] = d.xpos
        xquat[i] = d.xquat
    mujoco.mj_resetData(m, d)
    d.qpos[:] = 0.0
    d.qpos[0:3] = [0.0, 0.0, 0.8]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    np.savez_compressed(path, xpos=xpos, xquat=xquat, qpos=qpos.astype(np.float32),
                        rest_xpos=d.xpos.copy(), rest_xquat=d.xquat.copy(),
                        body_names=np.array(names, dtype=object), fps=FPS, mode=name)


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="moves_example.json")
    ap.add_argument("--only", default=None)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out-dir", default="out/moves")
    a = ap.parse_args()

    with open(a.spec) as fp:
        spec = json.load(fp)
    only = set(a.only.split(",")) if a.only else None
    os.makedirs(a.out_dir, exist_ok=True)

    demo, agent = build_stack()
    lib = load_pose_lib()
    gates = Gates(demo)

    for move in spec["moves"]:
        name = move["name"]
        if only and name not in only:
            continue
        print(f"\n=== {name} ({move['type']}) ===")
        results = []
        for seed in range(a.seeds):
            try:
                if move["type"] == "keyframes":
                    qpos, checkpoints, bounds = gen_keyframe_move(agent, lib, move, seed)
                    start = lib[move.get("start", "stance")][-1]
                    anchors = [(0, start[WRIST_COLS])] + \
                        [(fi, kq[WRIST_COLS]) for fi, kq in checkpoints]
                else:
                    qpos = gen_mode_move(agent, move["mode"], move.get("chunks", 3),
                                         move.get("dir", "none"), seed)
                    checkpoints, bounds = [], []
                    anchors = [(0, lib["stance"][-1][WRIST_COLS])]
                qpos = author_wrists(qpos, anchors)
                if move.get("loop"):
                    qpos, loop_err = loop_trim(qpos)
                    checkpoints = []
                g = gates.eval(qpos, checkpoints)
                g["frames"] = len(qpos)
                g["seed"] = seed
                g["score"] = score(g)
                results.append((g["score"], seed, qpos, g, bounds))
                print(f"  seed {seed}: score {g['score']:.4f}  kf_err {g['keyframe_ee_err']:.3f} "
                      f"skate {g['foot_skate_mean']:.3f}  jitter {g['jitter_mean']:.4f} "
                      f"frames {len(qpos)}")
            except Exception as e:
                print(f"  seed {seed}: FAILED {e}")

        if not results:
            print(f"  !! no successful seeds for {name}")
            continue
        results.sort(key=lambda r: r[0])
        _, best_seed, best_q, best_g, bounds = results[0]
        export_npz(demo, best_q, os.path.join(a.out_dir, f"{name}.npz"), name)

        meta = {"name": name, "best_seed": best_seed, "gates": best_g,
                "all_seeds": [r[3] for r in results],
                "segment_bounds": bounds, "fps": FPS}
        if move["type"] == "keyframes" and bounds and not move.get("loop"):
            meta["frame_data"] = {"startup": bounds[1] - 4,
                                  "active": [bounds[1] - 4, bounds[1]],
                                  "recovery": len(best_q) - bounds[1]}
        with open(os.path.join(a.out_dir, f"{name}.json"), "w") as fp:
            json.dump(meta, fp, indent=1)
        print(f"  -> best seed {best_seed} (score {best_g['score']:.4f}), "
              f"{len(best_q)} frames saved")


if __name__ == "__main__":
    main()
