import * as THREE from 'three';

// Rig-agnostic humanoid bone mapping.
//
// Resolves a skeleton's bones to canonical roles (Hips, Spine chain, Neck, Head,
// L/R Shoulder/Arm/ForeArm/Hand, L/R UpLeg/Leg/Foot/ToeBase) by name patterns
// covering Mixamo (with/without "mixamorig:"/"mixamorig_" prefix), Tripo3D's rig
// specs, VRM/UE-style names — with a topology fallback: if names don't match,
// classify by hierarchy shape (hips = lowest common ancestor of two leg chains
// and a spine chain, etc.).
//
// The output map always uses the CANONICAL names as keys; retarget.js works only
// with roles, so any rig this module can resolve is retargetable.

const CANON = [
  'Hips', 'Spine', 'Neck', 'Head',
  'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
  'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
  'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
  'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
];

// name → role patterns. Tried in order on a normalized name:
// lowercase, prefixes like "mixamorig:", "mixamorig_", "armature|" stripped,
// separators removed. Sides are detected separately (left/right/l_/_l/.l …).
const ROLE_PATTERNS = [
  // role, sided?, regexes (tested against the de-sided, normalized core name)
  ['Hips',      false, [/^(hips?|pelvis|root_?hips?|bip01pelvis|spinebase)$/]],
  ['Head',      false, [/^(head)$/]],
  ['Neck',      false, [/^(neck(twist)?0?[12]?)$/]],
  ['Shoulder',  true,  [/^(shoulder|clavicle|collar(bone)?)$/]],
  ['Arm',       true,  [/^(arm|upperarm|uparm|bicep)$/]],
  ['ForeArm',   true,  [/^(forearm|lowerarm|loarm|elbow)$/]],
  ['Hand',      true,  [/^(hand|wrist)$/]],
  ['UpLeg',     true,  [/^(upleg|upperleg|thigh|hip|leg_upper|upperlimb)$/]],
  ['Leg',       true,  [/^(leg|lowerleg|calf|shin|knee|leg_lower)$/]],
  ['Foot',      true,  [/^(foot|ankle)$/]],
  ['ToeBase',   true,  [/^(toebase|toes?|ball)$/]],
];

function normalize(name) {
  let n = name.toLowerCase();
  n = n.replace(/^.*[|]/, '');                    // "Armature|Hips" -> "hips"
  n = n.replace(/^(mixamorig|bip0?1|character\d*|b_|def[-_])[:._-]?/, '');
  n = n.replace(/^j_(bip|sec)[._-]?/, '');        // VRoid: J_Bip_C_Hips / J_Bip_L_UpperArm
  n = n.replace(/^c[._-]/, '');                   // VRoid center marker
  return n;
}

// returns {side: 'Left'|'Right'|null, core: string}
function splitSide(n) {
  let m;
  if ((m = n.match(/^(left|right)[._-]?(.*)$/))) return { side: m[1] === 'left' ? 'Left' : 'Right', core: m[2] };
  if ((m = n.match(/^(l|r)[._-](.*)$/))) return { side: m[1] === 'l' ? 'Left' : 'Right', core: m[2] };
  if ((m = n.match(/^(.*?)[._-]?(left|right)$/)) && m[2]) return { side: m[2] === 'left' ? 'Left' : 'Right', core: m[1] };
  if ((m = n.match(/^(.*)[._-](l|r)$/))) return { side: m[2] === 'l' ? 'Left' : 'Right', core: m[1] };
  return { side: null, core: n };
}

function stripSeps(s) { return s.replace(/[._\s-]/g, ''); }

// classify one bone name; returns canonical role name or null.
// spine handled separately (chains vary in length and naming).
function classify(name) {
  const n = normalize(name);
  const { side, core } = splitSide(n);
  const flat = stripSeps(core);
  for (const [role, sided, regs] of ROLE_PATTERNS) {
    for (const re of regs) {
      if (re.test(flat)) {
        if (sided && !side) {
          // "hip" without a side is not a leg; "arm"/"hand" without side ambiguous — skip
          return null;
        }
        return sided ? side + role : role;
      }
    }
  }
  return null;
}

function isSpineName(name) {
  const n = stripSeps(splitSide(normalize(name)).core);
  return /^(spine\d*|chest|upperchest|torso|waist)$/.test(n);
}

const wx = (b) => b.getWorldPosition(new THREE.Vector3());
const subtreeSize = (b) => { let n = 0; b.traverse(o => { if (o.isBone) n++; }); return n; };

// bones: array of THREE.Bone (any order). Returns:
// { map: {canonicalRole: boneName}, spineChain: [boneName bottom→top], ok, missing }
export function resolveRig(bones) {
  const byName = new Map(bones.map(b => [b.name, b]));
  const map = {};

  for (const b of bones) {
    const role = classify(b.name);
    if (role && !(role in map)) { map[role] = b.name; }
  }

  // ---- topological Hips fallback: first bone (root-down) with >=3 child subtrees,
  // or >=2 substantial ones (legs), below which the skeleton fans out
  if (!map.Hips) {
    const boneSet = new Set(bones);
    const roots = bones.filter(b => !b.parent || !boneSet.has(b.parent));
    const queue = [...roots];
    while (queue.length) {
      const b = queue.shift();
      const kids = b.children.filter(c => c.isBone);
      const substantial = kids.filter(k => subtreeSize(k) >= 3);
      if (substantial.length >= 3 || (substantial.length === 2 && kids.length >= 2)) {
        map.Hips = b.name; break;
      }
      queue.push(...kids);
    }
  }

  // ---- spine chain: walk from Hips toward Head/Neck, collecting spine-ish bones
  let spineChain = [];
  const hips = map.Hips && byName.get(map.Hips);
  if (hips) {
    // candidate starts: children of hips that are spine-named, or lead to the neck/arms
    const leadsToUpper = (bone) => {
      let found = false;
      bone.traverse(o => {
        if (!o.isBone) return;
        const r = classify(o.name);
        if (r === 'Neck' || r === 'Head' || r === 'LeftArm' || r === 'RightArm') found = true;
      });
      return found;
    };
    // a trunk bone is spine-named, or unclassified-but-leading-to-the-upper-body;
    // any bone that classifies to a concrete role (Neck, Shoulder, Arm…) ends the trunk
    const isTrunk = (c) => isSpineName(c.name) || (!classify(c.name) && leadsToUpper(c));
    let cur = hips.children.filter(c => c.isBone).find(isTrunk);
    while (cur) {
      if (classify(cur.name)) break;
      spineChain.push(cur.name);
      const kids = cur.children.filter(c => c.isBone);
      cur = kids.find(c => isSpineName(c.name)) || kids.find(c => !classify(c.name) && leadsToUpper(c));
      if (cur && spineChain.includes(cur.name)) break;
      if (spineChain.length > 8) break;
    }
  }
  // topological spine fallback: trunk = highest hips child; follow single-child
  // links; the first branch node is the chest
  if (hips && !spineChain.length) {
    const kids = hips.children.filter(c => c.isBone && subtreeSize(c) >= 2);
    const trunk = kids.length > 2 ? kids.reduce((a, b) => wx(a).y >= wx(b).y ? a : b) : null;
    let cur = trunk;
    while (cur) {
      spineChain.push(cur.name);
      const ck = cur.children.filter(c => c.isBone);
      if (ck.length !== 1 || spineChain.length > 8) break;
      cur = ck[0];
    }
  }
  // the top spine bone acts as the "chest" (arms/neck usually attach there);
  // if the arms attach lower in the chain, truncate to the attach bone
  if (spineChain.length) {
    map.Chest = spineChain[spineChain.length - 1];
    const armRef = map.LeftShoulder ?? map.LeftArm;
    if (armRef) {
      const attach = byName.get(armRef)?.parent;
      const i = attach ? spineChain.indexOf(attach.name) : -1;
      if (i >= 0 && i < spineChain.length - 1) {
        spineChain = spineChain.slice(0, i + 1);
        map.Chest = spineChain[i];
      }
    }
  }

  // ---- topology fallback for unresolved limb roles (works with arbitrary names)
  const missing = CANON.filter(r => !(r in map) && r !== 'LeftToeBase' && r !== 'RightToeBase'
    && r !== 'Neck' && r !== 'LeftShoulder' && r !== 'RightShoulder' && r !== 'Spine');
  if (hips && missing.length) topoFallback(bones, byName, map, spineChain);

  const still = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot',
    'LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm', 'RightHand']
    .filter(r => !(r in map));
  return {
    map, spineChain,
    ok: still.length === 0 && spineChain.length > 0,
    missing: still,
  };
}

// classify unresolved limbs by hierarchy + world position. Left/right comes
// from the feet's forward direction when toe bones exist (left = up × forward),
// falling back to "+x is the character's left" (three.js convention).
function topoFallback(bones, byName, map, spineChain) {
  const hips = byName.get(map.Hips);
  if (!hips) return;
  const spineSet = new Set(spineChain);

  const legRoots = hips.children.filter(c => c.isBone && !spineSet.has(c.name)
    && subtreeSize(c) >= 2 && wx(c).y <= wx(hips).y + 0.05);
  let leftDir = new THREE.Vector3(1, 0, 0);
  if (legRoots.length === 2 && !(map.LeftUpLeg && map.RightUpLeg)) {
    const [a, b] = legRoots;
    // provisional: A=left; correct using toe direction if available
    const chainA = chainOf(a);
    if (chainA.length >= 4) {
      const foot = chainA[2], toe = chainA[3];
      const fwd = wx(toe).sub(wx(foot)); fwd.y = 0;
      if (fwd.lengthSq() > 1e-6) {
        fwd.normalize();
        leftDir = new THREE.Vector3(0, 1, 0).cross(fwd).normalize();
      }
    }
    const sep = wx(a).sub(wx(b));
    const aIsLeft = sep.dot(leftDir) >= 0;
    assignChain(aIsLeft ? a : b, ['LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'], map);
    assignChain(aIsLeft ? b : a, ['RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase'], map);
  } else if (map.LeftUpLeg && map.RightUpLeg) {
    const l = byName.get(map.LeftUpLeg), r = byName.get(map.RightUpLeg);
    if (l && r) leftDir = wx(l).sub(wx(r)).normalize();
  }

  // chain completion: a chain root resolved by name but its children didn't
  // ("L_arm" matched, "L_fore" didn't) — finish the chain topologically
  for (const side of ['Left', 'Right']) {
    const arm = map[side + 'Arm'];
    if (arm && byName.get(arm) && (!map[side + 'ForeArm'] || !map[side + 'Hand'])) {
      assignChain(byName.get(arm), [side + 'Arm', side + 'ForeArm', side + 'Hand'], map);
    }
    const upleg = map[side + 'UpLeg'];
    if (upleg && byName.get(upleg) && (!map[side + 'Leg'] || !map[side + 'Foot'])) {
      assignChain(byName.get(upleg), [side + 'UpLeg', side + 'Leg', side + 'Foot', side + 'ToeBase'], map);
    }
  }

  const chest = map.Chest && byName.get(map.Chest);
  if (chest && !(map.LeftArm && map.RightArm)) {
    const cands = chest.children.filter(c => c.isBone)
      .map(c => ({ c, lat: wx(c).sub(wx(chest)).dot(leftDir), size: subtreeSize(c) }));
    const armRoots = cands.filter(x => x.size >= 3).sort((p, q) => q.lat - p.lat);
    if (armRoots.length >= 2) {
      assignArm(armRoots[0].c, 'Left', map);
      assignArm(armRoots[armRoots.length - 1].c, 'Right', map);
      // neck = the most central remaining child; its chain ends at the head
      if (!map.Neck) {
        const mid = cands.filter(x => x.c !== armRoots[0].c && x.c !== armRoots[armRoots.length - 1].c)
          .sort((p, q) => Math.abs(p.lat) - Math.abs(q.lat))[0];
        if (mid) {
          const nchain = chainOf(mid.c);
          map.Neck = nchain[0].name;
          if (!map.Head && nchain.length > 1) map.Head = nchain[nchain.length - 1].name;
        }
      }
    }
  }
}

function chainOf(root) {
  const out = [root];
  let cur = root;
  while (true) {
    const kids = cur.children.filter(c => c.isBone);
    if (!kids.length) break;
    // follow the longest sub-chain (main limb, not twist/roll helper bones)
    let best = null, bestLen = -1;
    for (const k of kids) {
      let len = 0; k.traverse(o => { if (o.isBone) len++; });
      if (len > bestLen) { bestLen = len; best = k; }
    }
    out.push(best); cur = best;
    if (out.length > 6) break;
  }
  return out;
}

function assignChain(root, roles, map) {
  const chain = chainOf(root);
  for (let i = 0; i < roles.length && i < chain.length; i++) {
    if (!(roles[i] in map)) map[roles[i]] = chain[i].name;
  }
}

function assignArm(root, side, map) {
  const chain = chainOf(root);
  const roles3 = [side + 'Arm', side + 'ForeArm', side + 'Hand'];
  const roles4 = [side + 'Shoulder', ...roles3];
  const roles = chain.length >= 4 ? roles4 : roles3;
  for (let i = 0; i < roles.length && i < chain.length; i++) {
    if (!(roles[i] in map)) map[roles[i]] = chain[i].name;
  }
}

export { CANON, classify, normalize };
