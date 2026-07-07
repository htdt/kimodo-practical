"""mbstack — thin glue over the MotionBricks demo stack.

Builds the full-agent inference stack (VQVAE + pose + root backbones + clip
holder) exactly like the interactive demo does, and wraps the deployed
generation step as a callable module. Everything here is extracted from the
demo utilities so that movegen.py / probe_api.py stay self-contained.

Run location: this file must sit inside GR00T-WholeBodyControl/motionbricks/
(see MOTIONBRICKS.md) so the relative checkpoint/asset paths resolve.
"""
import argparse
import os
import sys

import torch as t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motionbricks.motion_backbone.demo.clips import clip_holder_G1

DEVICE = 'cuda'
MODE_NAMES = list(clip_holder_G1.CLIPS.keys())


def build_args():
    """The navigation_demo argument set for headless batch generation
    (default checkpoints, no viewer, G1 clip holder)."""
    a = argparse.Namespace()
    a.result_dir = os.path.abspath("./out")
    a.data_root = os.path.abspath("./datasets")
    a.explicit_dataset_folder = None
    a.reprocess_clips = 0
    a.controller = "random"
    a.lookat_movement_direction = 0
    a.has_viewer = 0
    a.pre_filter_qpos = 1
    a.source_root_realignment = 1
    a.target_root_realignment = 1
    a.force_canonicalization = 1
    a.skip_ending_target_cond = 0
    a.random_speed_scale = 0
    a.speed_scale = [1.0, 1.0]
    a.generate_dt = 2.0
    a.disable_running = True
    a.new_control_dt = 2.0
    a.max_steps = 360
    a.random_seed = 1234
    a.num_runs = 1
    a.use_qpos = 1
    a.planner = "default"
    a.allowed_mode = None
    a.clips = "G1"
    a.return_model_configs = True
    a.return_dataloader = True
    a.recording_dir = None
    a.EXP = a.planner
    return a


class GenStep(t.nn.Module):
    """One full generation step of the deployed pipeline (canonicalize ->
    spring target model -> root transformer -> pose transformer -> VQVAE
    decode -> qpos), as a single forward call on CUDA."""

    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def forward(self, context_qpos, mode, movement_direction, facing_direction,
                random_seed, allowed_pred_num_tokens):
        a = self.agent
        inp = {
            'context_mujoco_qpos': context_qpos.clone(),
            'mode': mode,
            'movement_direction': movement_direction,
            'facing_direction': facing_direction,
            'random_seed': random_seed,
            'allowed_pred_num_tokens': allowed_pred_num_tokens,
        }
        inp['context_global_joint_positions'], inp['context_global_joint_rotations'] = \
            a._process_input_to_joint_transforms(inp)
        inp['target_root_position'], inp['target_root_positions'], \
            inp['target_root_headings'], inp['target_root_heading'], \
            inp['start_root_positions'], inp['start_root_headings'] = \
            a._generate_spring_model_position_and_heading(inp)
        inp['target_global_joint_positions'], inp['target_global_joint_rotations'], \
            inp['target_global_root_positions'] = a._generate_target_joint_transforms(inp)
        _, mujoco_qpos, num_pred_frames = a._generate_inbetween_frames(inp)
        return mujoco_qpos, num_pred_frames.view([1]).long()


def default_allowed(mode_idx):
    """The clip holder's allowed chunk-length mask for a mode, as a tensor."""
    v = clip_holder_G1.CLIPS[MODE_NAMES[mode_idx]].get('allowed_pred_num_tokens')
    if v is None:
        v = [1] * 11
    return t.tensor(v, dtype=t.long, device=DEVICE).view([1, -1])


def make_inputs(context_qpos, mode_idx, move_dir, face_dir, seed):
    return dict(
        context_qpos=context_qpos.float(),
        mode=t.tensor([[mode_idx]], dtype=t.long, device=DEVICE),
        movement_direction=t.tensor([move_dir], dtype=t.float32, device=DEVICE),
        facing_direction=t.tensor([face_dir], dtype=t.float32, device=DEVICE),
        random_seed=t.tensor([seed], dtype=t.long, device=DEVICE),
        allowed_pred_num_tokens=default_allowed(mode_idx),
    )
