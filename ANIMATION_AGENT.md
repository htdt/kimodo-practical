# ANIMATION_AGENT — choosing motion-authoring controls

Operational decision guide for an agent authoring moves through this
wrapper's move spec (`kimodo/kimogen.py`). Schemas, coordinate math, and
troubleshooting live in BAKE.md (spec + workflow), KIMODO.md (model,
conventions, gates), and ALIGN.md (rig certification) — this page only tells
you **which control to use when**. Every control listed here is implemented
and tested; `kimodo/test_constraints.py` validates this vocabulary.

## The controls

| control | move-spec field | use it for |
|---|---|---|
| **Text prompt** | `prompt` | motion intent: action, timing, mood, style — anywhere natural variation is acceptable |
| **Full-body target pose** | `constraints: [{"type":"fullbody", ...}]` | an exact key pose at a known frame: start/end bookends, in-betweening, matching another clip's pose |
| **Hand/foot end-effector target** | `left-hand` / `right-hand` / `left-foot` / `right-foot`, or `end-effector` + `joint_names` (exact tokens: `LeftHand`, `RightHand`, `LeftFoot`, `RightFoot`) | a spatial contact: grab, touch, plant, step, kick a known point, interact with an object — while the rest of the body stays free |
| **Root waypoints** | `root2d` with a few `frame_indices` | sparse navigation goals; Kimodo chooses the natural path between them |
| **Dense root path** | `root2d` with one entry per frame (+ optional `global_root_heading`) | a continuous trajectory that must follow a specific curve |
| **Combined constraints** | several objects in `constraints` | independent requirements from different families that must hold simultaneously |
| **Constraint-only generation** | omit `prompt` | spatially driven motion with no semantic text (supported by the installed model: an empty prompt is explicitly zeroed, never replaced by wording) |
| **`stance_bookend`** | `stance_bookend: true` | clips that must start and end in the shared stance — never duplicate those keyframes by hand |
| **Constraint file** | `constraints_file` (mutually exclusive with inline `constraints`) | reuse a JSON saved by the Kimodo demo/API verbatim |

## Selection rules

1. Use **text** for semantic intent. Never rely on wording for a measurable
   world-space requirement — gate it with a constraint instead.
2. Use the **narrowest constraint** that expresses the requirement: an
   end-effector target beats a full-body pose when only one hand or foot
   must hit a point.
3. Use a **full-body pose** only when the whole silhouette/joint arrangement
   matters.
4. Use **waypoints** for destinations; a **dense path** only when the entire
   route matters.
5. **Combine** controls only when each adds an independent requirement. The
   validator rejects duplicates and contradictions (same-frame double pins,
   root disagreements, impossible root speeds) — fix the conflict, never
   raise guidance weights blindly.
6. Keep sparse constraints **under 20 frames per type** (dense root paths
   exempt), keep post-processing on, and give the move enough `duration`
   for targets to be reachable (< 5 m/s root travel between pinned frames).
7. Vocabulary discipline: hand/foot **targets are authored constraints**;
   **foot contacts are model predictions** (QA evidence — and, for an
   authored foot key that lands inside a contact run, the span the runtime
   IK holds it over; never a target); unconstrained limbs are predictions.
   Don't describe the three as equivalent.
8. For every generated move, state which controls you selected and why,
   then check the generation report's constraint-adherence gates and the
   motion-quality gates before accepting
   (`kimogen.py report`, `qa_constraints.mjs`, `qa_endeffectors.mjs`).
9. **A prompt and its constraints are one statement at two resolutions.**
   When a pinned key pose changes, rewrite the prompt to match. A move whose
   keys say "sit back and kick the board sideways" while the prompt still
   says "drop into a deep crouch and put a hand on the road" meets every
   constraint and drifts back toward the old move in between them.
10. **A held shape is the hardest thing to ask for.** Grabs, guards,
   weapon-ready poses and tucks are silhouettes the whole move must keep,
   and the prior relaxes them wherever no key is watching. Pin them every
   ~15 frames rather than at the bookends and middle — and see "when the
   generator has an opinion" below when that is not enough.

## Authoring against a game that already owns its poses

The common case once a project is past prototype: the game has authored key
poses that pass its own stance/anatomy gate, and it needs *motion* between
them. Treat the split as **the game owns the character's relationship to its
prop; Kimodo owns the motion**, and the vocabulary above falls out:

- Text carries intent only. It cannot state "feet 0.54 m apart over the
  trucks with the shoulders along the deck", "the support hand stays on the
  fore-grip", "the guard stays up through the step" — and a move set authored
  from prompts alone will get that wrong in a *different* way each regenerate.
- `fullbody` keys carry the poses, because they are conditioned **and**
  corrected: a validated 7-move set held its authored shapes at 0.0000 m /
  0.00° on every pinned frame.
- Export per mapped joint the **global rotation delta from the character's
  bind pose**. That is directly a SOMA global rotation, because in Kimodo's
  output the T-pose is the zero pose (KIMODO.md, "the rest-pose trap") — so
  the game side needs no SOMA rest skeleton and the converter only composes
  globals into locals down the hierarchy: `local[j] = global[parent]⁻¹ · global[j]`.
- Do **not** also set `stance_bookend` on such a move: it pins frames
  `[0, T−1]`, which these specs already pin by hand, and the double pin is a
  validator error.

Three consequences, all of which only show up later:

**Adherence measures the keys, and says nothing about the in-betweens.**
Every real defect in that move set lived between keys, at 0.0000 m adherence
throughout: a launch that met every constraint exactly and *shook* between
them (jitter gate, 0.0253 against a 0.02 bound), and a held grab that hit
every key perfectly while the hips relaxed 0.26 m out of the arm's reach
halfway through, leaving the hand hovering over the rail. Run the consuming
game's own per-frame battery over the baked clip, held to the **union** of the
poses the clip was pinned to — a clip is a move, not a pose, and a
tuck-to-full-extension clip held to "full extension" alone fails on 15 frames
of 24 while doing exactly what was asked.

**When the generator has an opinion, denser keys are the wrong lever.** This
prior raises both arms over the head on a cruise and on both carves, in the
in-betweens where no keyframe can see it. Keys every 20 frames did not stop
it; every 13 did not stop it. What is wrong is not the shape at any key but
the model's idea of what the action involves, so the fix belongs in the
engine: let the clip own the body and have the game pull **one chain** back
toward the authored pose at partial weight (0.8 worked; the authored target
carries the game's own secondary motion so the limb still moves), and switch
the pull off where that limb's shape *is* the move. Expect to need this
wherever the model has a strong idea of its own — hands on a weapon, a
guard, a grip.

**And some moves should not be generated at all.** If the consuming runtime
places a clip-driven body by pinning a contact to a prop — feet to a deck,
hands to bars — then a move whose point is *breaking* that contact (a
superman off the board, a bail, a disarm) cannot be carried by a clip, no
matter how it is constrained. Author those in-engine and let entity motion (a
somersault, a dash) supply the movement. The same is true of any held
silhouette whose visible motion is entirely the entity's: generating it buys
drift and nothing else. Say so in the spec next to where the move would have
been, or the omission reads as a gap in the bake.

## Things the schema will enforce anyway

- Constraints are authored in Kimodo's **native frame**: Y-up, meters,
  heading 0 faces **+Z**, root starts near the XZ origin. (Accepted clips are
  canonicalized to +X later — the wrapper re-expresses your targets for you.)
- `frame_indices` are integers in `[0, duration·fps)`, sorted, unique.
- `fullbody` / end-effector constraints carry one **complete SOMA pose** per
  frame (`local_joints_rot` `[T,30|77,3]` axis-angle radians +
  `root_positions` `[T,3]`); the world target is the FK of that pose. An
  end-effector constraint **also pins the root XZ, root height, and heading**
  implied by its pose at those frames — co-framed constraints must agree.
- A `root2d` heading is a `[cos θ, sin θ]` pair per frame.
- Unreachable or contradictory authoring fails loudly, at validation when
  cheap (root speed, conflicts), else at the adherence gates.

## Minimal mixed example

```jsonc
{
  "name": "reach_while_walking",
  "duration": 3.0,
  "prompt": "A person walks forward and reaches for an object.",
  "constraints": [
    { "type": "root2d", "frame_indices": [0, 45, 89],
      "smooth_root_2d": [[0.0, 0.0], [0.4, 0.2], [0.9, 0.2]] },
    { "type": "right-hand", "frame_indices": [60],
      "local_joints_rot": [[ /* complete 30-joint axis-angle pose */ ]],
      "root_positions": [[0.55, 0.95, 0.2]] }
  ]
}
```

(Real complete pose arrays: see `kimodo/moveset_e2e.json`, generated from the
upstream demo poses by `make_e2e_spec.py`.)

## Input/output checklist

Before generating: frame indices in range and sorted · Y-up meters · native
+Z authoring frame · root speed between pins < 5 m/s · no same-frame double
pins · sparse frames per type < 20 · duration long enough to reach targets.

After generating: `accepted: true` in `out/moves/<move>.json` · adherence
gates green (`constraints.summary`: EE pos ≤ 5 mm, EE rot ≤ 2°, root XZ
≤ 2 cm) · `<move>.resolved_constraints.json` present (canonical targets for
runtime/QA) · then bake and run `qa_constraints.mjs <char> <movesDir> --gate`
per character.

Then, before the clip is allowed near the game: the **in-betweens**. Green
adherence plus a green `qa_constraints` run is necessary and not sufficient —
neither looks at whether the unpinned frames are still the move. If the
consuming game has a pose battery, run it per frame against the union of the
clip's pinned poses (see above); if it does not, that battery is the next
thing to write, not the clip after this one.
