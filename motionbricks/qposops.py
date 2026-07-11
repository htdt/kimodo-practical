"""qposops — pure-numpy qpos-space clip operations for the generation stage:
world-space ground-contact gates/fixes and the declarative `"post"` edit block
of the move spec (BAKE.md §3–4). Shared by movegen.py; importable anywhere.

Everything here operates on G1 mujoco qpos clips, shape (F, 36) =
root pos(3) + root quat WXYZ(4) + 29 dof, Z-up meters. FK is *injected*:
functions that need world ankle heights take an `ankle_z` array (F, 2)
(left/right world ankle Z per frame) or an `ankle_z_fn(qpos) -> (F, 2)`
callable, so this module has no mujoco dependency and self-tests with
synthetic kinematics:

  python qposops.py            # run the selftest battery (numpy only)

The selftest includes sabotage cases mirroring selftest.mjs: a clip whose
ankles pierce the floor must be caught by the ground gate (and healed by the
clamp), and a hop with no real flight must fail a `min_airborne` requirement.
"""
import sys

import numpy as np

# world-space gate constants (BAKE.md §4)
GROUND_EPS = -0.03   # min world ankle Z below this = ground penetration; reject
AIRBORNE_Z = 0.12    # both ankles above this = truly airborne (planted ankle
                     # centers sit at ~+0.04 on the G1 scene; generated jumps
                     # with real flight clear 0.12 for their whole air phase,
                     # grounded moves never do)

# wrist dof columns in qpos: L roll/pitch/yaw, R roll/pitch/yaw. These are
# AUTHORED control channels (BAKE.md §2d) — post edits never touch them.
WRIST_COLS = [26, 27, 28, 33, 34, 35]


# ------------------------------------------------------------ world-space gates

def ground_penetration(ankle_z):
    """Per-frame penetration depth below the floor plane, (F,) >= 0."""
    return np.clip(-np.asarray(ankle_z).min(axis=1), 0.0, None)


def airborne_frames(ankle_z, thresh=AIRBORNE_Z):
    """Longest consecutive run of frames with BOTH ankles above `thresh`."""
    both = (np.asarray(ankle_z) > thresh).all(axis=1)
    best = cur = 0
    for b in both:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return int(best)


def ground_lift(ankle_z, smooth=5):
    """Per-frame root lift that removes ground penetration: max ankle
    penetration per frame, smoothed (Hann window), then re-maxed with the raw
    penetration so smoothing can never leave residual penetration."""
    pen = ground_penetration(ankle_z)
    if smooth > 1 and len(pen) > 1:
        k = np.hanning(smooth + 2)[1:-1]
        sm = np.convolve(pen, k / k.sum(), mode="same")
        return np.maximum(pen, sm)
    return pen


def apply_ground_clamp(qpos, ankle_z, smooth=5):
    """Lift the root vertically by the smoothed per-frame penetration
    (the exact clamp used in the Lantern Rush field run, LIB_NOTES 2d)."""
    q = qpos.copy()
    q[:, 2] += ground_lift(ankle_z, smooth)
    return q


# ------------------------------------------------------------ post-edit ops

def envelope(F, a, b, peak):
    """Raised-cosine bump over frames [a, b] inclusive: 0 outside, `peak` at
    the window center, smooth ramps in and out (every frame in the window gets
    a non-zero weight)."""
    w = np.zeros(F)
    a, b = max(0, int(a)), min(F - 1, int(b))
    n = b - a + 1
    if n > 0:
        w[a:b + 1] = peak * np.hanning(n + 2)[1:-1]
    return w


def blend_to_pose(qpos, pose, frames, max_w=1.0, exclude_cols=WRIST_COLS):
    """Blend the joint dofs toward an authored pose (36,) over a frame window
    with a raised-cosine envelope peaking at `max_w`. Root pos/quat (cols 0:7)
    and excluded (authored) channels are never touched."""
    q = qpos.copy()
    w = envelope(len(q), frames[0], frames[1], max_w)
    cols = np.array([c for c in range(7, q.shape[1]) if c not in set(exclude_cols)])
    tgt = np.asarray(pose, dtype=q.dtype)[cols]
    q[:, cols] += w[:, None] * (tgt[None] - q[:, cols])
    return q


def yaw_twist(qpos, angle, frames):
    """Add a yaw rotation (radians, about world Z, in place about the pelvis)
    to the root orientation over a frame window, raised-cosine envelope —
    twists the body silhouette without moving the root path."""
    q = qpos.copy()
    half = envelope(len(q), frames[0], frames[1], float(angle)) / 2.0
    c, s = np.cos(half), np.sin(half)
    w2, x2, y2, z2 = q[:, 3:7].T.copy()       # copy: the assignments below alias q
    q[:, 3] = c * w2 - s * z2                 # (c,0,0,s) ⊗ root, WXYZ
    q[:, 4] = c * x2 - s * y2
    q[:, 5] = c * y2 + s * x2
    q[:, 6] = c * z2 + s * w2
    return q


def apply_post(qpos, post, pose_lib=None, ankle_z_fn=None, exclude_cols=WRIST_COLS):
    """Apply a move spec's declarative post block (BAKE.md §3) — a list of
    ops, in order, so cleanups survive every regeneration:

      {"blend_to_pose": "slide_lunge", "frames": [12, 28], "max_w": 0.9}
      {"yaw_twist": 0.62, "frames": [12, 28]}
      {"ground_clamp": true, "smooth": 5}

    pose_lib: {name: (4, 36) window} (blend uses the arrival frame [-1]);
    ankle_z_fn(qpos) -> (F, 2) world ankle heights (needed by ground_clamp).
    """
    q = qpos
    for op in post or []:
        if "blend_to_pose" in op:
            q = blend_to_pose(q, pose_lib[op["blend_to_pose"]][-1], op["frames"],
                              op.get("max_w", 1.0), exclude_cols=exclude_cols)
        elif "yaw_twist" in op:
            q = yaw_twist(q, op["yaw_twist"], op["frames"])
        elif "ground_clamp" in op:
            if op["ground_clamp"]:
                q = apply_ground_clamp(q, ankle_z_fn(q), op.get("smooth", 5))
        else:
            raise ValueError(f"unknown post op: {op}")
    return q


# ------------------------------------------------------------ selftest

def _selftest():
    failures = []

    def check(label, ok, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f"  — {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(label)

    rng = np.random.default_rng(7)
    F = 60

    def synth_qpos():
        q = np.zeros((F, 36))
        q[:, 2] = 0.8
        q[:, 3] = 1.0                                  # identity root quat
        q[:, 7:] = 0.1 * rng.standard_normal((F, 29))
        return q

    # synthetic FK: ankles ride the root height at fixed offsets
    base = np.stack([np.full(F, 0.04), np.full(F, 0.05)], axis=1)

    def fake_ankle_fn(offsets):
        return lambda q: offsets + (q[:, 2] - 0.8)[:, None]

    # --- ground gate: healthy clip passes, floor-piercing clip is caught ---
    check("ground gate: healthy clip has no penetration",
          float(base.min()) >= GROUND_EPS and ground_penetration(base).max() == 0.0)
    dip = base.copy()
    dip[20:30, 0] -= np.hanning(10) * 0.21             # ankle digs to ~-0.17 (field slide)
    check("sabotage ground gate: -0.17 m ankle dip is caught",
          float(dip.min()) < GROUND_EPS)

    # --- ground clamp heals the sabotage clip ---
    q = synth_qpos()
    fn = fake_ankle_fn(dip)
    qc = apply_ground_clamp(q, fn(q))
    check("ground clamp: no residual penetration", float(fn(qc).min()) >= -1e-9,
          f"min {float(fn(qc).min()):.4f}")
    check("ground clamp: clean frames untouched", np.allclose(qc[:14, 2], q[:14, 2]))
    pen = ground_penetration(fn(q))
    lift = qc[:, 2] - q[:, 2]
    first = int(np.argmax(pen > 0))
    check("ground clamp: lift is smoothed (ramps in before the dip)",
          lift[first - 1] > 0 and float(np.abs(np.diff(lift)).max()) < 0.09,
          f"lift[{first - 1}]={lift[first - 1]:.4f} max_step={np.abs(np.diff(lift)).max():.4f}")

    # --- airborne gate: real flight passes, ground-hugging hop fails ---
    jump = base.copy()
    jump[25:35] += 0.30                                # 10 frames of true flight
    check("airborne gate: real flight measured", airborne_frames(jump) == 10,
          f"got {airborne_frames(jump)}")
    hop = base.copy()
    hop[25:35, 0] += 0.30                              # one foot up, one planted
    hop[29:32, 1] += 0.10                              # other barely hops, below thresh
    check("sabotage airborne gate: ground-hugging hop fails min_airborne=6",
          airborne_frames(hop) < 6, f"got {airborne_frames(hop)}")

    # --- blend_to_pose: envelope blend, protected channels untouched ---
    q = synth_qpos()
    pose = np.zeros(36)
    pose[7:] = 1.0
    qb = blend_to_pose(q, pose, [20, 40], max_w=0.9)
    mid_err = np.abs(qb[30, 7:26] - pose[7:26]).mean()
    raw_err = np.abs(q[30, 7:26] - pose[7:26]).mean()
    check("blend_to_pose: mid-window pulled toward the pose", mid_err < 0.2 * raw_err,
          f"{mid_err:.3f} vs {raw_err:.3f}")
    check("blend_to_pose: outside the window untouched",
          np.allclose(qb[:20], q[:20]) and np.allclose(qb[41:], q[41:]))
    check("blend_to_pose: root + wrist channels never touched",
          np.allclose(qb[:, :7], q[:, :7]) and np.allclose(qb[:, WRIST_COLS], q[:, WRIST_COLS]))

    # --- yaw_twist: peak heading change ≈ angle, ends untouched, unit quats ---
    q = synth_qpos()
    qt = yaw_twist(q, 0.62, [12, 28])
    w, x, y, z = qt[:, 3], qt[:, 4], qt[:, 5], qt[:, 6]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    check("yaw_twist: peak twist ≈ authored angle", abs(yaw[20] - 0.62) < 0.05,
          f"peak {yaw[20]:.3f}")
    check("yaw_twist: outside the window untouched", np.allclose(qt[:12], q[:12])
          and np.allclose(qt[29:], q[29:]))
    check("yaw_twist: root quats stay unit",
          np.allclose(np.linalg.norm(qt[:, 3:7], axis=1), 1.0))

    # --- apply_post: ops compose in order; unknown op is an error ---
    lib = {"slide_lunge": np.tile(pose[None], (4, 1))}
    q = synth_qpos()
    fn = fake_ankle_fn(dip)
    qp = apply_post(q, [
        {"blend_to_pose": "slide_lunge", "frames": [20, 40], "max_w": 0.9},
        {"yaw_twist": 0.62, "frames": [20, 40]},
        {"ground_clamp": True},
    ], pose_lib=lib, ankle_z_fn=fn)
    check("apply_post: composed ops leave no penetration", float(fn(qp).min()) >= -1e-9)
    try:
        apply_post(q, [{"warp_time": 2.0}])
        check("apply_post: unknown op rejected", False)
    except ValueError:
        check("apply_post: unknown op rejected", True)

    print()
    if failures:
        print(f"{len(failures)} FAILURES")
        return 1
    print("qposops selftest passed")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
