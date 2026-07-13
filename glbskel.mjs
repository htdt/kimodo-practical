// Node-side GLB skeleton loader + animation sampler (no browser, no GLTFLoader).
// Builds a THREE.Bone hierarchy from a GLB's first skin via gltf-transform, with
// the armature's world transform baked into a wrapper Group so bone world
// positions match the file's world space (Meshy/Blender exports often carry a
// scaled Armature node above the joint root).
//
// Also samples embedded animation clips directly from the glTF samplers
// (LINEAR/STEP translation+rotation+scale), so an animated GLB can act as a
// motion SOURCE for the retargeter in tests — e.g. the Meshy walk clip.
import { NodeIO } from '@gltf-transform/core';
import * as THREE from 'three';

export async function loadGLBBones(path) {
  const doc = await new NodeIO().read(path);
  const root = doc.getRoot();
  const skin = root.listSkins()[0];
  if (!skin) throw new Error(`${path}: no skin`);
  const joints = skin.listJoints();
  if (!joints.length) throw new Error(`${path}: first skin has no joints`);
  const jointSet = new Set(joints);
  const names = new Set();

  const byNode = new Map();
  for (const j of joints) {
    if (!j.getName()) throw new Error(`${path}: every skin joint must have a name`);
    if (names.has(j.getName())) throw new Error(`${path}: duplicate skin joint name "${j.getName()}"`);
    names.add(j.getName());
    const b = new THREE.Bone();
    b.name = j.getName();
    b.position.set(...j.getTranslation());
    b.quaternion.set(...j.getRotation());
    b.scale.set(...j.getScale());
    byNode.set(j, b);
  }
  // parent joints to each other; non-joint parents (Armature) contribute a
  // baked world transform on a wrapper Group
  const wrapper = new THREE.Group();
  wrapper.name = 'glbskel_root';
  const topJoints = [];
  for (const j of joints) {
    const pj = j.listParents().find(p => jointSet.has(p));
    if (pj) byNode.get(pj).add(byNode.get(j));
    else { topJoints.push(j); wrapper.add(byNode.get(j)); }
  }
  if (topJoints.length !== 1)
    throw new Error(`${path}: first skin must have one joint root; found ${topJoints.length}`);
  const topJoint = topJoints[0];
  if (topJoint) {
    const pn = topJoint.listParents().find(p => p.propertyType === 'Node');
    if (pn) {
      const m = new THREE.Matrix4().fromArray(pn.getWorldMatrix());
      m.decompose(wrapper.position, wrapper.quaternion, wrapper.scale);
    }
  }
  wrapper.updateMatrixWorld(true);
  const bones = joints.map(j => byNode.get(j));
  const byName = new Map(bones.map(b => [b.name, b]));
  return { doc, bones, byName, byNode, wrapper, skin };
}

// Sample one animation clip: returns { duration, apply(t) } where apply(t)
// writes the sampled local TRS into the THREE.Bone hierarchy (byNode map) and
// refreshes world matrices. Non-joint channel targets are ignored.
export function animationSampler(doc, byNode, wrapper, clipIndex = 0) {
  const anims = doc.getRoot().listAnimations();
  if (!anims.length) throw new Error('no animations in GLB');
  if (!Number.isInteger(clipIndex) || clipIndex < 0 || clipIndex >= anims.length)
    throw new Error(`animation index ${clipIndex} out of range (0..${anims.length - 1})`);
  const anim = anims[clipIndex];
  const tracks = [];
  let duration = 0;
  for (const ch of anim.listChannels()) {
    const bone = byNode.get(ch.getTargetNode());
    if (!bone) continue;
    const s = ch.getSampler();
    const times = Array.from(s.getInput().getArray());
    const vals = s.getOutput().getArray();
    const path = ch.getTargetPath();       // translation | rotation | scale
    if (!['translation', 'rotation', 'scale'].includes(path)) continue;
    const stride = path === 'rotation' ? 4 : 3;
    const interpolation = s.getInterpolation();
    if (interpolation !== 'LINEAR' && interpolation !== 'STEP')
      throw new Error(`animation "${anim.getName()}" uses unsupported ${interpolation} interpolation`);
    if (!times.length || vals.length !== times.length * stride ||
        times.some((value, i) => !Number.isFinite(value) || (i > 0 && value <= times[i - 1])) ||
        Array.from(vals).some(value => !Number.isFinite(value)))
      throw new Error(`animation "${anim.getName()}" has malformed ${path} sampler data`);
    if (path === 'rotation') {
      for (let i = 0; i < vals.length; i += 4) {
        const norm2 = vals[i] ** 2 + vals[i + 1] ** 2 + vals[i + 2] ** 2 + vals[i + 3] ** 2;
        if (norm2 < 1e-12)
          throw new Error(`animation "${anim.getName()}" has a zero rotation quaternion`);
      }
    }
    const step = interpolation === 'STEP';
    duration = Math.max(duration, times[times.length - 1]);
    tracks.push({ bone, path, times, vals, stride, step });
  }
  if (!tracks.length) throw new Error(`animation "${anim.getName()}" has no skin-joint tracks`);
  const qa = new THREE.Quaternion(), qb = new THREE.Quaternion();
  function apply(t) {
    if (!Number.isFinite(t)) throw new Error(`animation sample time must be finite; got ${t}`);
    for (const tr of tracks) {
      const { bone, path, times, vals, stride, step } = tr;
      // locate segment (times are sorted, short arrays — linear scan is fine)
      let i = 0;
      while (i < times.length - 1 && times[i + 1] < t) i++;
      const i1 = Math.min(i + 1, times.length - 1);
      const t0 = times[i], t1 = times[i1];
      const a = (step || t1 <= t0) ? 0 : THREE.MathUtils.clamp((t - t0) / (t1 - t0), 0, 1);
      const o0 = i * stride, o1 = i1 * stride;
      if (path === 'rotation') {
        qa.set(vals[o0], vals[o0 + 1], vals[o0 + 2], vals[o0 + 3]).normalize();
        qb.set(vals[o1], vals[o1 + 1], vals[o1 + 2], vals[o1 + 3]).normalize();
        bone.quaternion.copy(qa.slerp(qb, a));
      } else {
        const target = path === 'translation' ? bone.position : bone.scale;
        target.set(
          vals[o0] + (vals[o1] - vals[o0]) * a,
          vals[o0 + 1] + (vals[o1 + 1] - vals[o0 + 1]) * a,
          vals[o0 + 2] + (vals[o1 + 2] - vals[o0 + 2]) * a);
      }
    }
    wrapper.updateMatrixWorld(true);
  }
  return { duration, apply, name: anim.getName() };
}
