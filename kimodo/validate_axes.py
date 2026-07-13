"""validate_axes — empirically pin Kimodo's axis/heading conventions.

Run on a generated 'walks forward' NPZ: travel direction must ≈ facing.
Prints heading[0], net root travel, and the hip-line-derived forward, and
checks the assumptions kimogen.py hardcodes:
  - Y-up (vertical variance lives in Y)
  - heading [cos t, sin t], facing dir = (sin t, 0, cos t)  [t=0 -> +Z]
  - ground at y=0 (foot joints touch ~0)
"""
import sys

import numpy as np

npz = sys.argv[1] if len(sys.argv) > 1 else "../kimodo_out/smoke_walk.npz"
z = np.load(npz)
print("keys:", list(z.keys()))
pj = z["posed_joints"]
print("posed_joints", pj.shape)

from kimodo.skeleton.definitions import SOMASkeleton30, SOMASkeleton77
sk = SOMASkeleton77() if pj.shape[1] == 77 else SOMASkeleton30()
print("skeleton:", sk.name)
names = sk.bone_order_names
idx = {n: i for i, n in enumerate(names)}

root = pj[:, sk.root_idx]
print(f"root y range: {root[:,1].min():.3f}..{root[:,1].max():.3f} (expect ~0.9 standing)")
feet = pj[:, [idx['LeftToeBase'], idx['RightToeBase']], 1]
print(f"toe min y: {feet.min():.3f} (expect ~0.0 = ground)")

travel = root[-1] - root[0]
print(f"net travel: x={travel[0]:+.3f} y={travel[1]:+.3f} z={travel[2]:+.3f}")

t0 = None
if "global_root_heading" in z:
    h = z["global_root_heading"]
    print(f"heading[0]: {h[0]} heading[-1]: {h[-1]}")
    t0 = np.arctan2(h[0][1], h[0][0])
    print(f"if [cos,sin]: t0={np.degrees(t0):.1f}deg -> facing (sin,cos)=({np.sin(t0):+.2f},{np.cos(t0):+.2f})")
else:
    print("(no global_root_heading key in this npz)")

# geometric forward at frame 0 from the hip line (right hip -> left hip x up)
r_hip, l_hip = pj[0, idx['RightLeg']], pj[0, idx['LeftLeg']]
right = r_hip - l_hip
up = np.array([0.0, 1.0, 0.0])
fwd_a = np.cross(right, up); fwd_a /= np.linalg.norm(fwd_a)
fwd_b = -fwd_a
tv = travel * np.array([1.0, 0.0, 1.0])
tvn = tv / (np.linalg.norm(tv) + 1e-9)
print(f"hip-line fwd candidate A (cross(right,up)): {fwd_a.round(2)} dot(travel)={fwd_a@tvn:+.2f}")
print(f"hip-line fwd candidate B (-A):              {fwd_b.round(2)} dot(travel)={fwd_b@tvn:+.2f}")
print("(walk-forward clip: the candidate with dot ~ +1 is the true forward;")
print(" compare with the heading-vector interpretations below)")
if t0 is not None:
    for lbl, v in [("(sin t0, cos t0)", np.array([np.sin(t0), 0, np.cos(t0)])),
                   ("(cos t0, sin t0)", np.array([np.cos(t0), 0, np.sin(t0)]))]:
        print(f"heading as {lbl}: {v.round(2)} dot(travel)={v@tvn:+.2f}")
