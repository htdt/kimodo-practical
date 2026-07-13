"""kimogen — MK move-set generation on NVIDIA Kimodo (SOMA skeleton).

The Stage 2 generation step (BAKE.md §3-4): each move is
text-prompted (combat is in Kimodo's training distribution), optionally
bookended with a fighting-stance fullbody keyframe constraint at both ends
(so clips chain/crossfade in-game), generated as a best-of-N batch, and
pushed through numeric QA gates. Winner NPZ + gate/frame-data JSON per move.

Usage (venv: kimenv; start `TEXT_ENCODER_DEVICE=cpu kimodo_textencoder` first
or let it fall back to an in-process CPU encoder):

  python kimogen.py gen   --spec moveset_mk.json [--only jab,sweep] [--samples 8] [--seed 42]
  python kimogen.py stance                    # extract stance pose from out/moves/idle_stance.npz
  python kimogen.py report                    # gate table of everything generated so far

Axis conventions (Kimodo): Y-up, XZ ground, heading angle t about Y with
facing dir (sin t, cos t) — i.e. heading 0 faces +Z. Verified by
validate_axes.py on a generated walk. Gates evaluate in a canonical frame
(frame-0 root at origin, frame-0 facing rotated to +X = the baked-clip/game
convention).
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MOVES_OUT = os.path.join(OUT, "moves")
STANCE_PATH = os.path.join(OUT, "stance_pose.json")

# somaskel77 indices are resolved at runtime from the skeleton class
EE_NAMES = ["LeftHand", "RightHand", "LeftFoot", "RightFoot", "Head"]
CONTACT_JOINTS = ["LeftFoot", "LeftToeBase", "LeftToeEnd",
                  "RightFoot", "RightToeBase", "RightToeEnd"]
CONTACT_JOINTS_4 = ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]
STANCE_MAX_ERROR = 0.22
FOOT_SKATE_MAX = 0.12
JITTER_MAX = 0.015
SAFE_MOVE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


# ---------------------------------------------------------------- geometry --
def canonicalize(joints, root_pos, idx):
    """Rotate/translate so frame-0 root sits at XZ origin facing +X.

    Frame-0 facing is derived from the hip line: fwd = cross(up, rHip-lHip)
    (verified on Kimodo outputs by validate_axes.py — walk-forward clips
    travel along this vector). Returns (joints', root', R).
    """
    if len(joints) == 0 or len(root_pos) != len(joints):
        raise ValueError("motion must contain equally sized, non-empty joint and root arrays")
    right = joints[0, idx["RightLeg"]] - joints[0, idx["LeftLeg"]]
    fwd = np.cross(np.array([0.0, 1.0, 0.0]), right)
    fwd[1] = 0.0
    fwd_len = np.linalg.norm(fwd)
    if not np.isfinite(fwd_len) or fwd_len < 1e-6:
        raise ValueError("cannot derive heading: frame-0 hip line is degenerate")
    fwd /= fwd_len
    c, s = fwd[0], fwd[2]                     # rot_y mapping fwd -> +X
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    p0 = root_pos[0] * np.array([1.0, 0.0, 1.0])
    j = (joints - p0) @ R.T
    r = (root_pos - p0) @ R.T
    return j, r, R


# ------------------------------------------------------------------- gates --
def failed_gates(reason, nonfinite=False):
    """Consistent diagnostic record for a sample that cannot be measured."""
    return {"nonfinite": bool(nonfinite), "malformed": reason,
            "foot_skate_mean": 1e9, "contact_frac": 0.0,
            "contact_ok": False, "foot_skate_ok": False,
            "jitter_mean": 1e9, "jitter_ok": False,
            "travel_x": 0.0, "travel_z": 0.0, "travel_ok": False,
            "apex_ok": False, "stance_ok": False,
            "pass": False, "score": 1e30}


def gate_sample(j, r, fc, idx, mv, stance, fps):
    """All-numeric QA for one canonicalized sample. Returns dict of metrics."""
    g = {}
    if fc.ndim != 2 or fc.shape[0] != len(j) or fc.shape[1] not in (4, 6):
        return failed_gates(f"foot_contacts shape {fc.shape}")
    g["nonfinite"] = bool(not np.isfinite(j).all() or not np.isfinite(r).all()
                          or not np.isfinite(fc).all())
    if g["nonfinite"]:
        return failed_gates("motion contains non-finite values", nonfinite=True)

    # foot skate: horizontal speed of contact-labeled foot joints (m/s)
    contact_names = CONTACT_JOINTS if fc.shape[1] == 6 else CONTACT_JOINTS_4
    cj = [idx[n] for n in contact_names]
    v = np.linalg.norm(np.diff(j[:, cj][:, :, [0, 2]], axis=0), axis=-1) * fps
    c = (fc[1:] > 0.5) & (fc[:-1] > 0.5)
    g["foot_skate_mean"] = float(v[c].mean()) if c.any() else 0.0
    g["contact_frac"] = float((fc > 0.5).any(axis=-1).mean())
    g["contact_ok"] = g["contact_frac"] >= 0.05 and bool(c.any())
    g["foot_skate_ok"] = g["foot_skate_mean"] <= FOOT_SKATE_MAX

    # jitter: mean 2nd difference of joint positions (m/frame^2)
    g["jitter_mean"] = float(np.linalg.norm(np.diff(j, n=2, axis=0), axis=-1).mean())
    g["jitter_ok"] = g["jitter_mean"] <= JITTER_MAX

    # net travel along the fight axis (+X = toward opponent)
    g["travel_x"] = float(r[-1, 0] - r[0, 0])
    g["travel_z"] = float(abs(r[-1, 2] - r[0, 2]))

    tv = mv.get("travel")
    if tv == "in_place":
        g["travel_ok"] = abs(g["travel_x"]) < 0.45 and g["travel_z"] < 0.45
    elif tv == "fwd":
        g["travel_ok"] = g["travel_x"] > 0.3 and g["travel_z"] < 0.6
    elif tv == "back":
        g["travel_ok"] = g["travel_x"] < -0.15 and g["travel_z"] < 0.6
    else:
        g["travel_ok"] = True

    # apex checks (root rise/dip, strike height, floor contact)
    apex = mv.get("apex")
    if apex:
        y0 = r[0, 1]
        if apex["kind"] == "root_rise":
            g["apex_val"] = float(r[:, 1].max() - y0)
            g["apex_ok"] = g["apex_val"] >= apex["min"]
        elif apex["kind"] == "root_dip":
            g["apex_val"] = float(y0 - r[:, 1].min())
            g["apex_ok"] = g["apex_val"] >= apex["min"]
        elif apex["kind"] == "root_floor":
            g["apex_val"] = float(r[:, 1].min())
            g["apex_ok"] = g["apex_val"] <= apex["max"]
        elif apex["kind"] == "ankle_height":
            ank = max(float(j[:, idx["LeftFoot"], 1].max()),
                      float(j[:, idx["RightFoot"], 1].max()))
            g["apex_val"] = ank
            g["apex_ok"] = ank >= apex["min"]
    else:
        g["apex_ok"] = True

    # stance match at both ends (root-relative EE positions vs stance ref)
    if stance is not None and mv.get("stance_bookend"):
        ee = [idx[n] for n in EE_NAMES]
        ref = np.asarray(stance["ee_root_rel"])              # [5,3]
        def enderr(f):
            cur = j[f, ee] - r[f] * np.array([1.0, 0.0, 1.0])
            return float(np.linalg.norm(cur - ref, axis=-1).mean())
        g["stance_err_start"] = enderr(0)
        g["stance_err_end"] = enderr(-1)
        g["stance_ok"] = (g["stance_err_start"] <= STANCE_MAX_ERROR
                          and g["stance_err_end"] <= STANCE_MAX_ERROR)
    else:
        g["stance_ok"] = True

    g["pass"] = (not g["nonfinite"]) and g["contact_ok"] and g["foot_skate_ok"] \
        and g["jitter_ok"] and g["travel_ok"] and g["apex_ok"] and g["stance_ok"]
    # score: lower better; only meaningful among passing samples
    g["score"] = (g["foot_skate_mean"] * 2.0 + g["jitter_mean"] * 30.0
                  + (g.get("stance_err_start", 0.0)
                     + g.get("stance_err_end", 0.0)) * 1.5)
    return g


def frame_data(j, r, idx, mv, fps):
    """startup/active/recovery from strike-limb tip speed (game frame data)."""
    if not mv.get("strike"):
        return None
    tips = ["LeftHand", "RightHand"] if mv["strike"] == "hand" else \
           ["LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"]
    sp = [np.linalg.norm(np.diff(j[:, idx[n]], axis=0), axis=-1) * fps for n in tips]
    sp = np.stack(sp)                                        # [tips, T-1]
    tip = int(sp.max(axis=1).argmax())
    s = sp[tip]
    pk = int(s.argmax())
    thr = max(0.6 * s[pk], 1.0)                              # m/s
    a = pk
    while a > 0 and s[a - 1] > thr * 0.5:
        a -= 1
    b = pk
    while b < len(s) - 1 and s[b + 1] > thr * 0.5:
        b += 1
    active = [int(a), int(min(b + 2, len(s)))]
    # contact = the frame the strike visually lands: max extension of the
    # striking tip from the root inside the active window. The active window
    # alone is not enough for impact timing — it opens while the limb is
    # still travelling (generated motion has real wind-up), so events synced
    # to active[0] fire visibly early. Games should register the hit no
    # earlier than ~contact-2 and one-shot effects (damage, sfx, hitstop)
    # exactly at contact.
    ext = np.linalg.norm(j[:, idx[tips[tip]]] - r, axis=-1)
    contact = active[0] + int(ext[active[0]:active[1] + 1].argmax())
    return {"startup": int(a), "active": active, "contact": int(contact),
            "recovery": int(len(s) + 1 - active[1]),
            "strike_tip": tips[tip], "peak_speed": round(float(s[pk]), 2),
            "height": mv.get("height")}


def best_loop(j, r, min_len, max_len):
    """Find (i0, i1) trimming to the best pose-space cycle for loop moves."""
    T = len(j)
    root_rel = j - r[:, None, :] * np.array([1.0, 0.0, 1.0])
    if T < 2:
        raise ValueError("a loop needs at least two frames")
    min_len = max(1, min(int(min_len), T - 1))
    max_len = max(min_len, min(int(max_len), T - 1))
    best, pair = float("inf"), (0, T - 1)
    for i in range(0, T - min_len):
        jmax = min(T - 1, i + max_len)
        for k in range(i + min_len, jmax + 1):
            d = float(np.linalg.norm(root_rel[i] - root_rel[k], axis=-1).mean())
            if d < best:
                best, pair = d, (i, k)
    return pair, best


# --------------------------------------------------------------- generation --
def load_kimodo(modelname):
    import torch
    from kimodo import load_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(modelname, device=device, default_family="Kimodo")
    return model


def build_bookend_constraints(stance, nF):
    """Fullbody stance keyframes at the first and last frames (somaskel30)."""
    rot = stance["local_joints_rot30"]
    rp = stance["root_pos"]
    return [{
        "type": "fullbody",
        "frame_indices": [0, nF - 1],
        "local_joints_rot": [rot, rot],
        "root_positions": [[0.0, rp[1], 0.0], [0.0, rp[1], 0.0]],
    }]


def gen_move(model, mv, stance, samples, seed, steps):
    from kimodo.constraints import load_constraints_lst
    from kimodo.tools import seed_everything

    npz_path = os.path.join(MOVES_OUT, f"{mv['name']}.npz")
    report_path = os.path.join(MOVES_OUT, f"{mv['name']}.json")
    constraint_path = os.path.join(MOVES_OUT, f"{mv['name']}.constraints.json")
    # A regeneration attempt owns these outputs. Clear prior accepted files up
    # front so an exception or interrupted run cannot silently leave stale
    # motion available to the bake step.
    for stale in (npz_path, report_path, constraint_path):
        if os.path.exists(stale):
            os.remove(stale)

    fps = model.fps
    nF = int(mv["duration"] * fps)
    constraint_lst = []
    cfg_kwargs = {}
    if stance is not None and mv.get("stance_bookend"):
        with open(constraint_path, "w") as fp:
            json.dump(build_bookend_constraints(stance, nF), fp)
        constraint_lst = load_constraints_lst(constraint_path, model.skeleton)
        cfg_kwargs = {"cfg_type": "separated", "cfg_weight": [2.0, 2.0]}

    seed_everything(seed)
    out = model([mv["prompt"]], [nF],
                constraint_lst=constraint_lst,
                num_denoising_steps=steps,
                num_samples=samples,
                multi_prompt=True,
                post_processing=True,
                return_numpy=True,
                **cfg_kwargs)

    required_out = {"posed_joints", "root_positions", "foot_contacts",
                    "global_rot_mats", "local_rot_mats"}
    missing_out = required_out - set(out)
    if missing_out:
        raise RuntimeError("Kimodo output is missing " + ", ".join(sorted(missing_out)))
    posed = np.asarray(out["posed_joints"])
    roots = np.asarray(out["root_positions"])
    contacts = np.asarray(out["foot_contacts"])
    global_rots = np.asarray(out["global_rot_mats"])
    local_rots = np.asarray(out["local_rot_mats"])
    if (posed.ndim != 4 or posed.shape[0] < 1 or posed.shape[1] < 3
            or posed.shape[2:] != (77, 3)
            or roots.shape != (posed.shape[0], posed.shape[1], 3)
            or contacts.ndim != 3 or contacts.shape[:2] != posed.shape[:2]
            or contacts.shape[2] not in (4, 6)
            or global_rots.shape != (*posed.shape[:3], 3, 3)
            or local_rots.shape != (*posed.shape[:3], 3, 3)):
        raise RuntimeError("Kimodo returned inconsistent motion array shapes")

    skel77 = model.skeleton.somaskel77
    idx = {n: i for i, n in enumerate(skel77.bone_order_names)}
    root_i = skel77.root_idx

    results = []
    for s in range(posed.shape[0]):
        pj, rp, fc = posed[s], roots[s], contacts[s]
        try:
            if not np.isfinite(global_rots[s]).all() or not np.isfinite(local_rots[s]).all():
                raise ValueError("rotation channels contain non-finite values")
            jc, rc, R = canonicalize(pj, rp, idx)
            g = gate_sample(jc, rc, fc, idx, mv, stance, fps)
        except (ValueError, IndexError) as error:
            jc = rc = R = None
            g = failed_gates(f"canonicalization failed: {error}")
        results.append((g, s, jc, rc, R))
    if not results:
        raise RuntimeError(f"Kimodo returned no samples for {mv['name']}")

    passing = [t for t in results if t[0]["pass"]]
    if not passing:
        g, s, _, _, _ = min(results, key=lambda t: t[0]["score"])
        report = {"name": mv["name"], "picked_sample": int(s),
                  "num_samples": len(results), "num_passing": 0,
                  "accepted": False, "loop": bool(mv.get("loop")),
                  "gates": {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in g.items()},
                  "all_gates": [{k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in t[0].items()} for t in results],
                  "frame_data": None,
                  "frames": int(posed[s].shape[0]), "fps": int(fps)}
        with open(report_path, "w") as fp:
            json.dump(report, fp, indent=1, allow_nan=False)
        return report
    g, s, jc, rc, R = min(passing, key=lambda t: t[0]["score"])

    # loop trim on the canonical winner
    trim = None
    if mv.get("loop"):
        (i0, i1), loop_err = best_loop(jc, rc, int(1.0 * fps), int(3.2 * fps))
        trim = (i0, i1)
        g["loop_err"] = round(loop_err, 4)
        g["loop_trim"] = [i0, i1]

    sl = slice(trim[0], trim[1] + 1) if trim else slice(None)
    jc_clip, rc_clip, trim_R = canonicalize(jc[sl], rc[sl], idx)
    total_R = trim_R @ R
    grm = np.einsum("ij,tnjk->tnik", total_R, global_rots[s][sl])
    lrm = local_rots[s][sl].copy()
    lrm[:, root_i] = np.einsum("ij,tjk->tik", total_R, lrm[:, root_i])

    np.savez_compressed(
        npz_path,
        posed_joints=jc_clip.astype(np.float32),
        root_positions=rc_clip.astype(np.float32),
        global_rot_mats=grm.astype(np.float32),
        local_rot_mats=lrm.astype(np.float32),
        foot_contacts=contacts[s][sl].astype(np.float32),
        canonical_rotation=total_R.astype(np.float32),
        fps=fps)

    fd = frame_data(jc_clip, rc_clip, idx, mv, fps)
    report = {"name": mv["name"], "picked_sample": int(s),
              "num_samples": len(results), "num_passing": len(passing),
              "accepted": True, "loop": bool(mv.get("loop")),
              "gates": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in g.items()},
              "all_gates": [{k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in t[0].items()} for t in results],
              "frame_data": fd, "frames": int(jc_clip.shape[0]), "fps": int(fps)}
    with open(report_path, "w") as fp:
        json.dump(report, fp, indent=1, allow_nan=False)
    return report


# ------------------------------------------------------------------ stance --
def extract_stance():
    """Pick the most representative frame of idle_stance as THE stance pose."""
    import torch
    from kimodo.skeleton.definitions import SOMASkeleton30, SOMASkeleton77

    with np.load(os.path.join(MOVES_OUT, "idle_stance.npz")) as z:
        required = {"posed_joints", "root_positions", "local_rot_mats", "canonical_rotation"}
        missing = required - set(z.files)
        if missing:
            raise RuntimeError("idle_stance.npz is missing " + ", ".join(sorted(missing))
                               + "; regenerate it")
        j, r, lrm = z["posed_joints"], z["root_positions"], z["local_rot_mats"]
        canonical_rotation = z["canonical_rotation"]
    if (j.ndim != 3 or j.shape[1:] != (77, 3) or len(j) < 2
            or r.shape != (len(j), 3) or lrm.shape != (len(j), 77, 3, 3)
            or canonical_rotation.shape != (3, 3)
            or not np.isfinite(j).all() or not np.isfinite(r).all()
            or not np.isfinite(lrm).all() or not np.isfinite(canonical_rotation).all()):
        raise RuntimeError("idle_stance.npz has invalid motion shapes or values; regenerate it")
    sk30, sk77 = SOMASkeleton30(), SOMASkeleton77()
    idx = {n: i for i, n in enumerate(sk77.bone_order_names)}

    # medoid frame in root-relative pose space = the pose the idle keeps returning to
    rel = j - r[:, None, :] * np.array([1.0, 0.0, 1.0])
    D = np.linalg.norm(rel[:, None] - rel[None], axis=-1).mean(-1)
    f = int(D.mean(1).argmin())

    lrm30 = sk30.from_SOMASkeleton77(torch.from_numpy(lrm[f][None]))[0].numpy()
    # Constraints are authored in Kimodo's native frame. Undo the exact
    # canonicalization used for this generated (and possibly loop-trimmed)
    # clip; a hard-coded 90° undo loses the trim frame's heading correction.
    lrm30[sk30.root_idx] = canonical_rotation.T @ lrm30[sk30.root_idx]
    from scipy.spatial.transform import Rotation as Rot
    aa = Rot.from_matrix(lrm30).as_rotvec()

    ee = [idx[n] for n in EE_NAMES]
    stance = {
        "frame": f,
        "local_joints_rot30": np.round(aa, 5).tolist(),
        "root_pos": np.round(r[f], 4).tolist(),
        "ee_root_rel": np.round(j[f, ee] - r[f] * np.array([1.0, 0.0, 1.0]), 4).tolist(),
    }
    with open(STANCE_PATH, "w") as fp:
        json.dump(stance, fp)
    print(f"[stance] frame {f} of idle_stance -> {STANCE_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "stance", "report"])
    ap.add_argument("--spec", default=os.path.join(HERE, "moveset_mk.json"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--model", default="Kimodo-SOMA-RP-v1.1")
    a = ap.parse_args()
    os.makedirs(MOVES_OUT, exist_ok=True)

    if a.cmd == "stance":
        return extract_stance()

    if a.cmd == "report":
        for f in sorted(os.listdir(MOVES_OUT)):
            if f.endswith(".json") and not f.endswith("constraints.json"):
                with open(os.path.join(MOVES_OUT, f)) as fp:
                    rep = json.load(fp)
                g = rep["gates"]
                print(f"{rep['name']:<14} pass={g.get('pass')} "
                      f"skate={g.get('foot_skate_mean')} jitter={g.get('jitter_mean')} "
                      f"travel_x={g.get('travel_x')} apex={g.get('apex_val', '-')} "
                      f"stance_end={g.get('stance_err_end', '-')} "
                      f"({rep['num_passing']}/{rep['num_samples']} passing)")
        return

    with open(a.spec) as fp:
        spec = json.load(fp)
    moves = spec.get("moves")
    if not isinstance(moves, list) or not moves:
        ap.error("spec must contain a non-empty 'moves' array")
    names = [mv.get("name") for mv in moves]
    if any(not isinstance(n, str) or not SAFE_MOVE_NAME.fullmatch(n) for n in names):
        ap.error("move names must contain only letters, numbers, '_' and '-'")
    if len(set(names)) != len(names):
        ap.error("move names must be unique")
    if any(not isinstance(mv.get("prompt"), str) or not mv["prompt"].strip()
           or isinstance(mv.get("duration"), bool)
           or not isinstance(mv.get("duration"), (int, float))
           or not 0 < mv["duration"] <= 10 for mv in moves):
        ap.error("every move needs a non-empty prompt and a duration in (0, 10] seconds")
    valid_travel = {None, "in_place", "fwd", "back"}
    valid_apex = {"root_rise": "min", "root_dip": "min",
                  "root_floor": "max", "ankle_height": "min"}
    for mv in moves:
        if mv.get("travel") not in valid_travel:
            ap.error(f"{mv['name']}: travel must be in_place, fwd, back, or null")
        if "loop" in mv and not isinstance(mv["loop"], bool):
            ap.error(f"{mv['name']}: loop must be boolean")
        if "stance_bookend" in mv and not isinstance(mv["stance_bookend"], bool):
            ap.error(f"{mv['name']}: stance_bookend must be boolean")
        if mv.get("strike") not in (None, "hand", "foot"):
            ap.error(f"{mv['name']}: strike must be hand or foot")
        apex = mv.get("apex")
        if apex is not None:
            key = valid_apex.get(apex.get("kind")) if isinstance(apex, dict) else None
            value = apex.get(key) if key else None
            if (key is None or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value < 0):
                ap.error(f"{mv['name']}: invalid apex definition")
        if "height" in mv and mv["height"] not in ("low", "mid", "high"):
            ap.error(f"{mv['name']}: height must be low, mid, or high")
    if a.samples < 1 or a.steps < 1:
        ap.error("--samples and --steps must be positive")
    only = set(a.only.split(",")) if a.only else None
    unknown = only - set(names) if only else set()
    if unknown:
        ap.error("--only contains unknown moves: " + ", ".join(sorted(unknown)))
    selected = [mv for mv in moves if not only or mv["name"] in only]
    if os.path.exists(STANCE_PATH):
        with open(STANCE_PATH) as fp:
            stance = json.load(fp)
    else:
        stance = None
    if stance is not None:
        try:
            rot = np.asarray(stance["local_joints_rot30"], dtype=float)
            root = np.asarray(stance["root_pos"], dtype=float)
            ee = np.asarray(stance["ee_root_rel"], dtype=float)
        except (KeyError, TypeError, ValueError):
            ap.error("out/stance_pose.json is malformed; regenerate it with 'stance'")
        if (rot.shape != (30, 3) or root.shape != (3,) or ee.shape != (5, 3)
                or not np.isfinite(rot).all() or not np.isfinite(root).all()
                or not np.isfinite(ee).all()):
            ap.error("out/stance_pose.json is malformed; regenerate it with 'stance'")
    # The unqualified second pass means "everything else": do not overwrite
    # the idle clip from which the active stance constraint was extracted.
    # An explicit idle-only pass remains the way to replace that source.
    has_bookends = any(mv.get("stance_bookend") for mv in selected)
    if stance is not None and a.only is None and has_bookends:
        selected = [mv for mv in selected if mv["name"] != "idle_stance"]
    elif (stance is not None and has_bookends
          and any(mv["name"] == "idle_stance" for mv in selected)):
        ap.error("generate idle_stance separately before moves that use its stance")
    if stance is None and any(mv.get("stance_bookend") for mv in selected):
        ap.error("bookended moves require out/stance_pose.json; generate idle_stance, then run 'stance' first")

    model = load_kimodo(a.model)
    if "fps" in spec and int(spec["fps"]) != int(model.fps):
        ap.error(f"spec fps={spec['fps']} does not match model fps={model.fps}")
    too_short = [mv["name"] for mv in selected if int(mv["duration"] * model.fps) < 3]
    if too_short:
        ap.error("moves must produce at least 3 frames: " + ", ".join(too_short))
    failed = []
    for mv in selected:
        if mv["name"] == "idle_stance" and os.path.exists(STANCE_PATH):
            os.remove(STANCE_PATH)
            stance = None
            print("[stance] removed stale stance_pose.json; run 'stance' after idle generation")
        print(f"[gen] {mv['name']}: '{mv['prompt'][:60]}...' "
              f"({mv['duration']}s x{a.samples})")
        rep = gen_move(model, mv, stance, a.samples, a.seed, a.steps)
        g = rep["gates"]
        print(f"      -> accepted={rep['accepted']} ({rep['num_passing']}/{rep['num_samples']} passing) "
              f"skate={g['foot_skate_mean']} travel_x={g['travel_x']}")
        if not rep["accepted"]:
            failed.append(mv["name"])
    if failed:
        print("[reject] no passing sample for: " + ", ".join(failed), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
