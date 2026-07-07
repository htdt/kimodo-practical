// Standalone self-test — NO external assets. Proves the module ships: builds
// two synthetic humanoids from scratch (different naming families, different
// bind facings, different proportions, one with a 0.01-scaled armature),
// procedurally animates the source, and runs the full alignment certification
// source→target, plus a mirrored-map sabotage that must FAIL.
// Usage: node selftest.mjs   (needs only node_modules: three)
import * as THREE from 'three';
import {
  rigFromBones, srcMapFromRig, snapshotMotion, mineProbeFrames,
  certifyRig, checkSideConsistency, resetBindPose,
} from './align.js';

let failures = 0;
function check(label, ok, detail = '') {
  if (ok) console.log(`ok   ${label}`);
  else { failures++; console.log(`FAIL ${label}${detail ? '  — ' + detail : ''}`); }
}

// build a bone hierarchy from { name: [parentName, [x,y,z]] } (parents first);
// wraps in a Group with the given uniform scale (bone offsets given in WORLD
// units are divided by it, like scaled-armature exports store them)
function buildRig(spec, armatureScale = 1) {
  const bones = {};
  const wrapper = new THREE.Group();
  wrapper.scale.setScalar(armatureScale);
  for (const [name, [parent, pos]] of Object.entries(spec)) {
    const b = new THREE.Bone();
    b.name = name;
    b.position.set(...pos).multiplyScalar(1 / armatureScale);
    bones[name] = b;
    if (parent) bones[parent].add(b); else wrapper.add(b);
  }
  // child local offsets were divided once too much (parents already carry the
  // scale correction exactly once, at the wrapper) — restore all but the root
  for (const [name, [parent]] of Object.entries(spec)) {
    if (parent) bones[name].position.multiplyScalar(armatureScale);
  }
  wrapper.updateMatrixWorld(true);
  return { bones, wrapper, all: Object.values(bones) };
}

// SOURCE: UE-style names, binds facing +X (left limbs at −Z), height ~1.5
// (offsets are parent-relative; world = accumulated)
const L = -0.09, R = +0.09;                       // lateral z for a +X-facing rig
const srcSpec = {
  pelvis: [null, [0, 0.86, 0]],
  spine_01: ['pelvis', [0, 0.14, 0]],
  spine_02: ['spine_01', [0, 0.14, 0]],
  spine_03: ['spine_02', [0, 0.14, 0]],
  neck_01: ['spine_03', [0, 0.10, 0]],
  head: ['neck_01', [0, 0.09, 0]],
  clavicle_l: ['spine_03', [0.01, 0.04, L * 0.5]], upperarm_l: ['clavicle_l', [0, 0, L]],
  lowerarm_l: ['upperarm_l', [0, 0, L * 2.4]], hand_l: ['lowerarm_l', [0, 0, L * 2.2]],
  clavicle_r: ['spine_03', [0.01, 0.04, R * 0.5]], upperarm_r: ['clavicle_r', [0, 0, R]],
  lowerarm_r: ['upperarm_r', [0, 0, R * 2.4]], hand_r: ['lowerarm_r', [0, 0, R * 2.2]],
  thigh_l: ['pelvis', [0, -0.04, L]], calf_l: ['thigh_l', [0, -0.40, 0]],
  foot_l: ['calf_l', [0, -0.38, 0]], ball_l: ['foot_l', [0.12, -0.04, 0]],
  thigh_r: ['pelvis', [0, -0.04, R]], calf_r: ['thigh_r', [0, -0.40, 0]],
  foot_r: ['calf_r', [0, -0.38, 0]], ball_r: ['foot_r', [0.12, -0.04, 0]],
};

// TARGET: Mixamo-style names, binds facing +Z (left limbs at +X), DIFFERENT
// proportions (longer legs, shorter torso, wider shoulders), 0.01 armature
const tgtSpec = {
  Hips: [null, [0, 1.02, 0]],
  Spine: ['Hips', [0, 0.16, 0]],
  Spine1: ['Spine', [0, 0.16, 0]],
  Spine2: ['Spine1', [0, 0.16, 0]],
  Neck: ['Spine2', [0, 0.10, 0]],
  Head: ['Neck', [0, 0.10, 0]],
  LeftShoulder: ['Spine2', [0.07, 0.04, 0]], LeftArm: ['LeftShoulder', [0.13, 0, 0]],
  LeftForeArm: ['LeftArm', [0.27, 0, 0]], LeftHand: ['LeftForeArm', [0.25, 0, 0]],
  RightShoulder: ['Spine2', [-0.07, 0.04, 0]], RightArm: ['RightShoulder', [-0.13, 0, 0]],
  RightForeArm: ['RightArm', [-0.27, 0, 0]], RightHand: ['RightForeArm', [-0.25, 0, 0]],
  LeftUpLeg: ['Hips', [0.11, -0.05, 0]], LeftLeg: ['LeftUpLeg', [0, -0.48, 0]],
  LeftFoot: ['LeftLeg', [0, -0.44, 0]], LeftToeBase: ['LeftFoot', [0, -0.05, 0.13]],
  RightUpLeg: ['Hips', [-0.11, -0.05, 0]], RightLeg: ['RightUpLeg', [0, -0.48, 0]],
  RightFoot: ['RightLeg', [0, -0.44, 0]], RightToeBase: ['RightFoot', [0, -0.05, 0.13]],
};

const src = buildRig(srcSpec);
const srcT = rigFromBones(src.all);
check('source rig resolves (UE names, +X facing)', srcT.rig.ok, 'missing=' + srcT.rig.missing);

const tgt = buildRig(tgtSpec, 0.01);
const tgtT = rigFromBones(tgt.all);
check('target rig resolves (Mixamo names, +Z facing, 0.01 armature)', tgtT.rig.ok,
  'missing=' + tgtT.rig.missing);
check('target world scale sane despite armature', Math.abs(
  tgt.bones.Hips.getWorldPosition(new THREE.Vector3()).y - 1.02) < 1e-6);

// procedurally animate the source: torso twist, arm swings + elbow bends,
// leg swings + knee bends, slight pelvis bob — moderate, humanoid-plausible
const N = 90, FPS = 30;
const basePelvisY = src.bones.pelvis.position.y;
function pose(f) {
  const t = f / FPS;
  const s = (w, ph = 0) => Math.sin(2 * Math.PI * w * t + ph);
  src.bones.pelvis.position.y = basePelvisY - 0.05 * (1 - Math.cos(2 * Math.PI * t)) / 2;
  src.bones.spine_02.rotation.y = 0.30 * s(0.5);
  src.bones.upperarm_l.rotation.x = 0.55 * s(1);          // swing about the facing axis
  src.bones.upperarm_r.rotation.x = -0.55 * s(1, Math.PI / 3);
  src.bones.lowerarm_l.rotation.y = 0.45 * (1 - Math.cos(2 * Math.PI * t)) / 2;
  src.bones.lowerarm_r.rotation.y = -0.45 * (1 - Math.cos(2 * Math.PI * t + 1)) / 2;
  src.bones.hand_l.rotation.z = 0.30 * s(1.5);
  src.bones.thigh_l.rotation.z = 0.45 * s(1);              // hip flexion about lateral (z)
  src.bones.thigh_r.rotation.z = -0.45 * s(1);
  src.bones.calf_l.rotation.z = -0.35 * (1 - Math.cos(2 * Math.PI * t)) / 2;
  src.bones.calf_r.rotation.z = -0.35 * (1 + Math.cos(2 * Math.PI * t)) / 2;
  src.bones.foot_l.rotation.z = 0.15 * s(1, 1);
  src.wrapper.updateMatrixWorld(true);
}
const motion = snapshotMotion(srcT.orderedBones, pose, N, FPS, 'selftest');
check('motion snapshot has quats + rest', Array.isArray(motion.quat) && motion.rest.length === srcT.orderedBones.length);

const srcMap = srcMapFromRig(srcT.rig.map);
const probes = mineProbeFrames([motion], srcMap);
check('probes mined from synthetic clip', probes.numFrames === 7, `got ${probes.numFrames}`);

// full certification source→target
const cert = certifyRig(tgtT, probes, { srcMap });
check('synthetic source→target certifies', cert.pass, cert.failures.join('; '));
check('  side consistency verified', cert.gates.sideConsistency === true);
check('  bone stretch ~0', cert.gates.boneStretchPct < 0.1, `${cert.gates.boneStretchPct}%`);
check('  round trip tight', cert.gates.roundTripMean < 0.05 && cert.gates.roundTripP95 < 0.10,
  `mean ${cert.gates.roundTripMean} p95 ${cert.gates.roundTripP95}`);
console.log('     gates:', JSON.stringify(cert.gates));

// sabotage: mirrored legs must FAIL certification via the absolute side gate
{
  const t2 = rigFromBones(buildRig(tgtSpec, 0.01).all);
  const m = t2.rig.map;
  for (const r of ['UpLeg', 'Leg', 'Foot', 'ToeBase']) {
    [m['Left' + r], m['Right' + r]] = [m['Right' + r], m['Left' + r]];
  }
  const side = checkSideConsistency(t2);
  check('sabotage mirrored legs: side gate fires', side !== null && !side.ok);
  const cert2 = certifyRig(t2, probes, { srcMap });
  check('sabotage mirrored legs: certification FAILS', !cert2.pass);
}

resetBindPose(tgtT);
console.log(failures ? `\n${failures} FAILURES` : '\nselftest passed (no external assets used)');
process.exit(failures ? 1 : 0);
