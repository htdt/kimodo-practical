#!/usr/bin/env node
// Alignment certification CLI (pipeline Stage 1, see ALIGN.md).
// Usage: node certify.mjs <char.glb> --clips a.json,b.json [--srcmap map.json] [--out cert.json]
// Clips are motion JSONs in the format documented in ALIGN.md; probe poses are
// mined from them deterministically. The source role map is taken from
// --srcmap, else from a `srcMap` field on the first clip, else defaults to the
// SOMA skeleton (SOMA_SRC). Writes <char.glb>.retarget_certificate.json
// next to the GLB unless --out is given.
// Exit code 0 = certified, 1 = gate failure, 2 = invalid input or rig/tool error.
import fs from 'node:fs';
import path from 'node:path';
import { loadGLBBones } from './glbskel.mjs';
import { rigFromBones, mineProbeFrames, certifyRig } from './align.js';

const args = process.argv.slice(2);
const known = new Set(['clips', 'srcmap', 'out']);
const opts = Object.create(null), positional = [];
let parseError = null;
for (let i = 0; i < args.length && !parseError; i++) {
  const arg = args[i];
  if (!arg.startsWith('--')) { positional.push(arg); continue; }
  const name = arg.slice(2);
  if (!known.has(name)) { parseError = `unknown option --${name}`; continue; }
  if (Object.hasOwn(opts, name)) { parseError = `duplicate option --${name}`; continue; }
  const value = args[++i];
  if (!value || value.startsWith('--')) { parseError = `--${name} requires a value`; continue; }
  opts[name] = value;
}
const glb = positional.length === 1 ? positional[0] : null;
const clipsArg = opts.clips;
if (parseError || !glb || !clipsArg) {
  if (parseError) console.error(parseError);
  console.error('usage: node certify.mjs <char.glb> --clips a.json,b.json [--srcmap map.json] [--out cert.json]');
  process.exit(2);
}
const clipPaths = clipsArg.split(',').map(p => p.trim()).filter(Boolean);
if (!clipPaths.length) {
  console.error('--clips must contain at least one path');
  process.exit(2);
}
const srcmapPath = opts.srcmap;
const out = opts.out ?? glb + '.retarget_certificate.json';
const protectedPaths = [glb, ...clipPaths, ...(srcmapPath ? [srcmapPath] : [])]
  .map(p => path.resolve(p));
if (protectedPaths.includes(path.resolve(out))) {
  console.error('--out must not overwrite the character, a clip, or the source map');
  process.exit(2);
}

let clips, srcMap, probes, target;
try {
  clips = clipPaths.map(p => JSON.parse(fs.readFileSync(p, 'utf8')));
  srcMap = srcmapPath ? JSON.parse(fs.readFileSync(srcmapPath, 'utf8'))
    : clips[0].srcMap ?? undefined;
  probes = mineProbeFrames(clips, srcMap);
  const { bones } = await loadGLBBones(glb);
  target = rigFromBones(bones);
} catch (e) {
  console.error(JSON.stringify({ ok: false, error: e.message }));
  process.exit(2);
}

let cert;
try {
  cert = certifyRig(target, probes, srcMap ? { srcMap } : {});
} catch (e) {
  console.error(JSON.stringify({ ok: false, error: e.message }));
  process.exit(2);
}
const doc = {
  ok: cert.pass,
  character: path.resolve(glb),
  generatedAt: new Date().toISOString(),
  probeClips: clipPaths.map(p => path.basename(p)),
  ...cert,
};
try {
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, JSON.stringify(doc, null, 2));
} catch (e) {
  console.error(JSON.stringify({ ok: false, error: e.message }));
  process.exit(2);
}
console.log(JSON.stringify({
  ok: cert.pass, certificate: out,
  gates: cert.gates, failures: cert.failures,
  rig: { rolesResolved: cert.rig.rolesResolved, missing: cert.rig.missing },
}, null, 2));
process.exit(cert.pass ? 0 : 1);
