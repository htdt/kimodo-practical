"""Install smoke-test: load the full MotionBricks stack, verify keyframe-API
shapes and qpos layout. Run once after MOTIONBRICKS.md setup:
  xvfb-run -a python probe_api.py
Success = the prints below with no exception (first run compiles/loads
checkpoints and takes a minute)."""
import os, sys
import torch as t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from mbstack import build_args, DEVICE
from motionbricks.motion_backbone.demo.utils import navigation_demo

demo = navigation_demo(build_args())
agent = demo.full_agent
conv = agent._converter
mrep = agent._motion_rep
inf = agent._inferencer

print("device:", DEVICE, "cuda avail:", t.cuda.is_available())
print("fps:", mrep.fps)
print("CLIPS:", list(agent._clip_holder.CLIPS.keys()))
print("num_frames_per_clip:", agent._clip_holder.num_frames_per_clip.tolist())

qpos = agent._clip_holder.mujoco_qpos[0, :4][None]  # idle first 4 frames
print("qpos shape:", qpos.shape)
pos, rot = conv.convert_mujoco_qpos_to_motion_transforms(qpos.to(DEVICE))
print("transforms:", pos.shape, rot.shape)  # [1,4,J,3], [1,4,J,3,3]
J = pos.shape[2]
print("J =", J, " local_poses dim =", (J - 1) * 3 + J * 6)

rootnet = inf._root_model.backbone_net
print("min_tokens:", rootnet._args['min_tokens'], "max_tokens:", rootnet._args['max_tokens'],
      "MASKED:", rootnet.MASKED_NUM_TOKENS)

# mujoco model
import mujoco
m = demo.mj_model
print("nq:", m.nq, "nbody:", m.nbody, "timestep fps:", 1.0 / m.opt.timestep)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
print("bodies:", names)
jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
print("joints:", jnames)
print("jnt_range:", m.jnt_range[:5], "...")
print("\nPROBE OK — MotionBricks stack is functional.")
