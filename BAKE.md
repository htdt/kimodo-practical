# Stage 2 — BAKE: author, generate, gate, and bake motion clips

Everything in this stage happens on the **canonical skeleton** — the skeleton
of your motion source, not of any character. Clips authored here drive every
certified character forever; a new character never requires regenerating a
clip, and a new clip never requires touching a character.

The idea in one line: **author sparse keyframe poses, let your motion source
generate the movement between them, gate the results numerically, keep the
best of N attempts, and bake canonicalized clips + frame data.**

```
pose library          move spec              generation             bake
mine/author    ──►    keyframe schedule ──►  N seeds per move  ──►  canonicalize,
keyframe poses        (JSON, per move)       → QA gates             frame data,
(verified by eye)                            → best-of-N            manifest
```

## 0. Prerequisites

- A motion source. The validated path is a **keyframe-conditioned generative
  inbetweener** (a model that takes context frames + a target keyframe window
  and generates the motion between). Alternatives that fit the same pipeline:
  cutting clips from a mocap library, a text-to-motion model, hand-keyed
  animation — anything that can emit the clip format below.
- A **certified character** to preview on (Stage 1, `certify.mjs` — see
  ALIGN.md). Never evaluate motion on an uncertified rig: you can't tell
  motion defects from transfer defects.

## 1. The clip contract

Every baked clip is a motion JSON in the ALIGN.md format (world-space joint
positions + quaternions per frame, Y-up meters, with `rest`/`restQuat`), and
**canonicalized**: frame 0 pelvis at the origin, heading = 0, canonical
forward = **+X**. Canonicalize at bake time — raw generations and mocap cuts
inherit arbitrary world frames, and the runtime (Stage 3) depends on every
clip agreeing on origin and forward.

Root motion stays **in the clip data** (the pelvis actually travels). The
runtime decides how to consume it; never bake clips "in place".

Alongside the clips, emit a `manifest.json`: the ordered list of moves, each
with its file, frame count, fps, loop flag, and frame data (§6).

## 2. Build a pose library

A keyframe is a full-body pose on the canonical skeleton. Store poses as
small **windows** (e.g. 4 consecutive frames — whatever your generator's
constraint API takes), in two flavors:

- a **moving window** (consecutive frames from a real clip) keeps the pose's
  momentum — right for strike apexes and anything the motion should *swing
  through*;
- a **held pose** (one frame tiled) has zero velocity — right for stances,
  guards, crouches: poses the motion should *arrive at and stop*.

### 2a. Mine from mocap (preferred)

Scan whatever mocap you have with cheap heuristics (highest foot = kick apex,
lowest pelvis = crouch, widest hand span = reach, …) to shortlist candidate
frames, then **render a contact sheet and look at it before saving anything**
— heuristics shortlist, your eyes decide. Save the chosen frames under
semantic names (`kick_high`, `stance`, `hit_recoil`).

### 2b. Author novel poses (fallback)

For poses no mocap has: copy the nearest library pose, edit joint angles,
render, iterate — cap it at ~3 rounds; if it's still wrong, the pose is
probably outside your generator's reachable set. Stay inside the source
skeleton's joint limits.

### 2c. Overshoot deliberately

Generative priors are conservative — they undershoot amplitude. Keyframes
*pull* the generation toward them, so author apex poses at or slightly past
what you actually want (head-height ankle, full lunge). If generation can't
reach the pose (arrival-error gate fails on every seed), back the pose off.

### 2d. Unregularized channels are authored, never generated

Rule learned the hard way: any channel your generator doesn't regularize
(for a robot-skeleton source that was the **wrists** — tiny links, barely
represented in the model's features, output was 40°+/frame noise) must be
**discarded from the generated data and rebuilt** as smooth interpolation
between explicitly authored per-keyframe values. Filtering noise leaves
smoothed noise; authored control channels make flicker impossible by
construction. If such a channel ever looks wrong, the fix is a pose-library
edit and a regenerate — never a runtime filter.

## 3. Write the move spec

One JSON spec per move set. Two move types:

```jsonc
{"moves": [
  // keyframe-driven (attacks, reactions, poses)
  {"name": "uppercut", "type": "keyframes",
   "start": "stance",              // context pose the move begins from
   "steps": [                      // each step = one generation chunk
     {"pose": "crouch_deep", "frames": 24},
     {"pose": "strike_rising", "frames": 24},
     {"pose": "stance", "frames": 24}       // ALWAYS return to stance
   ]},
  // native-skill rollout (locomotion — the model's own prior is the source)
  {"name": "walk_fwd", "type": "mode", "mode": "walk", "dir": "fwd",
   "chunks": 4, "loop": true}
]}
```

Useful per-step fields: target pose, duration (pin it, or let the model
predict), root displacement `dxy` (for lunges and knockback), heading change.
Move-level `loop: true` means "trim the result to its best pose-space cycle"
(idles, walks).

Design rules that survived a full move-set in production:

- **Bookend every move with the same stance pose.** Start from it, end on it.
  This is what lets clips chain and crossfade in-game with a single short
  blend.
- **One action per step.** `stance → kick → stance`, never
  `stance → kick_and_recover`. Use an intermediate held pose (a deep crouch
  inside an uppercut) to shape the path.
- **Respect your generator's minimum chunk length.** If the shortest natural
  generation is ~24 frames and the game needs a 12-frame jab, generate 24 and
  play it faster in-engine (Stage 3 handles this) — don't fight the prior.
- **Ground-contact changes are fine** (stance → flying hit → on the ground →
  get up) but give those steps extra duration.

## 4. Generate with best-of-N

Per move, run every seed (N = 8–16; make sampling stochastic so seeds
actually differ) through the whole keyframe schedule, gate each candidate,
keep the best. Generation is typically seconds per move on a consumer GPU —
seeds are cheap, debugging a bad clip downstream is not.

| gate | meaning | healthy | reject |
|---|---|---|---|
| keyframe arrival error | mean end-effector distance (wrists/ankles/torso, root-relative) between the generated arrival frame and the keyframe | 0.03–0.06 m | > 0.1 m = keyframe not reached |
| foot skate | mean horizontal ankle speed during ground contact | source-prior level | rising above it |
| jitter | mean 2nd difference of joint angles | ≤ 0.03 rad | visible vibration |
| limit violations | fraction of frames outside joint range | 0.0 | any |

**Gate what you actually care about, in world space.** A cautionary tale: a
"jump" selected by these gates alone never left the ground — foot-skate gates
*reward* staying planted, and a pelvis-relative arrival error can't see root
height. The fix was a keyframe through a mined airborne pose plus a direct
check on pelvis/ankle world trajectories. When a move has a defining physical
property (airborne, floor contact, displacement), assert it explicitly.

**The reject loop.** Keyframe unreachable on all seeds → reduce overshoot or
add an intermediate step. Duration feels wrong → pin a different length. One
seed does something weird → more seeds. Always verify visually with a
stick-figure filmstrip *on the canonical skeleton* before baking — cheaper
than debugging through the retargeter.

## 5. Bake

Canonicalize (§1), compute per-clip metadata, write clips + `manifest.json`.
Keep the baked clips character-agnostic; retargeting happens at load/run time
through the certified retargeter (or, equivalently, pre-bake per-character
engine-native clips — glTF animations, engine `AnimationClip`s — by running
the retargeter offline once per character and exporting; the runtime cost is
identical to hand-authored animation either way).

## 6. Frame data (for gameplay)

Derive combat/gameplay timing from the generation itself — never hand-author
it:

- **startup** = frames until the first keyframe arrival (the apex),
- **active** = a small window around that arrival,
- **recovery** = the rest.

Sanity-check against strike-limb tip velocity peaks. Store per move in the
manifest. Stage 3 builds hit detection, reach, and interruption rules purely
from this data.

## 7. Visual QA rubric

Certification (Stage 1) catches *rig* problems; this catches *motion*
problems. Render every move on a certified character (contact sheet + video)
and check:

- keyframe apex actually visible and at the intended amplitude?
- feet planted during stances/guards (no skate)?
- no wrist flips or knee pops at chunk seams?
- root travel matches intent (lunge forward, knockback back, in-place else)?

Weak poses are **pose-library edits + regenerate** (~minutes), never runtime
patches.

## 8. New character = zero work here

Clips know nothing about characters. For a new model: certify it (Stage 1),
then play the existing clips through its certified retarget. Regeneration is
only ever needed for new *moves*.
