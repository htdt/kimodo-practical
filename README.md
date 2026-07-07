# skeleton-align — animate any humanoid character, in any engine

A manual (plus the alignment scripts it is built around) for implementing
character animation as a **bake pipeline**: motion is authored or generated
once on a canonical skeleton, quality-gated, retargeted onto a **certified**
character rig, and shipped as ordinary baked clips. The game runtime contains
zero neural code and zero hand-tuned motion constants.

Written to be followed end-to-end by an autonomous coding agent (or a human).
It is engine-agnostic and game-agnostic: the reference implementation is
three.js, and it has been validated by building a complete two-player fighting
game from generated motion, but every stage states its contract in plain data
(JSON formats, gate thresholds, state-machine shapes) that ports anywhere.

## The pipeline

```
 Stage 1 — ALIGN            Stage 2 — BAKE                Stage 3 — INTEGRATE
 certify the rig            author clips on the           play baked clips:
 (rigmap roles, probe       canonical skeleton:           state machine, root-
 battery, gates,            keyframes → generation →      motion integration,
 certificate JSON)          QA gates → best-of-N →        crossfades, frame-data
                            canonicalize + frame data     gameplay, QA harness
        │                          │                              │
   ALIGN.md                    BAKE.md                      INTEGRATE.md
```

Two principles carry the whole design:

1. **Don't try to make retargeting always right — make it impossible to ship
   wrong.** Stage 1 is a certification battery, not a fixed mapping: absolute
   gates + round-trip metrics, pass-or-reject. An uncertified character never
   enters animation production.
2. **Clips are character-agnostic.** Everything in Stage 2 targets the
   canonical skeleton, not a character. A new character costs one
   certification run (seconds); a new move costs one generation run. The two
   never multiply.

## What's in this repo

| File | What it is |
|---|---|
| `ALIGN.md` | Stage 1 manual: rig resolution, retargeting, certification battery |
| `BAKE.md` | Stage 2 manual: pose libraries, move specs, generation gates, baking |
| `INTEGRATE.md` | Stage 3 manual: runtime layers, root motion, combat timing, QA |
| `rigmap.js` | bone → canonical-role resolution for arbitrary humanoid rigs |
| `retarget.js` | the two-skeleton position-based retargeter |
| `align.js` | probe mining, inverse recovery, gates, `certifyRig` |
| `glbskel.mjs` | GLB → bone hierarchy + animation sampler in node (no browser) |
| `certify.mjs` | certification CLI, writes `<char.glb>.retarget_certificate.json` |
| `selftest.mjs` | zero-asset self-test (synthetic rigs, procedural motion, sabotage case) |

The scripts are the complete Stage 1 implementation and the runtime retarget
layer used in Stage 3. Stage 2's generation half is deliberately **not**
code in this repo: it depends on your motion source (a generative model, a
mocap library, hand keying). BAKE.md specifies the contracts — input pose
format, gate thresholds, output clip format — that any source must meet.

## Quick start

```bash
npm install        # three + @gltf-transform/core, nothing else
npm test           # selftest: full synthetic certification, no assets needed
```

Certify a character against motion clips (format in ALIGN.md):

```bash
node certify.mjs character.glb --clips walk.json,kick.json
# → character.glb.retarget_certificate.json, exit 0 = certified
```

Retarget any motion source onto any humanoid GLB in the browser:

```js
import { loadGLBSkeleton, buildBoneOrder, Retargeter } from './retarget.js';

const { hips } = await loadGLBSkeleton(GLTFLoader, './character.glb', scene);
const orderedBones = buildBoneOrder(hips);
const bones = {}; orderedBones.forEach(b => bones[b.name] = b);
const rt = new Retargeter({ bones, orderedBones, hips, hipsParent: hips.parent, data: motion });
rt.applyFrame(f);   // pose the rig for frame f, call per render tick
```

## For agents: how to use this repo

You were probably sent here to add animated characters to a game. The order
of work is fixed and each stage gates the next:

1. Read **ALIGN.md**. Certify every character rig first. If certification
   fails, fix or regenerate the rig — do not proceed with an uncertified
   character; every downstream artifact would be built on a broken mapping.
2. Read **BAKE.md**. Author the move set on the canonical skeleton, gate every
   generated clip numerically, then look at filmstrips before accepting.
   Output: canonicalized clips + a manifest + frame data.
3. Read **INTEGRATE.md**. Wire the clips into the game with the three-layer
   architecture (clip / entity / game). Build the deterministic QA harness
   *before* tuning gameplay — it is what makes the rest debuggable.

Throughout: prefer regenerating a failed artifact over patching around it at
runtime; every gate threshold in these docs was validated in practice — treat
a gate failure as a real defect, not noise.

## Assumptions & limits

- **Humanoids only**: two legs, two arms, one spine chain. Toes, shoulders and
  neck are optional and degrade gracefully; missing core roles are a hard fail.
- **Y-up, meters**, feet flat and pointing forward at bind pose. Uniform
  armature scale (0.01-scaled FBX→GLB exports are handled).
- Verified against Mixamo, Tripo3D, Meshy AI and UE-style rigs, plus the
  Unitree G1 robot skeleton as a motion source; runs in the browser and node.

## License

MIT
