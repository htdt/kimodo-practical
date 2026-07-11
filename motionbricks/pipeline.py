"""pipeline — one driver for the whole rebake ritual (BAKE.md §5, INTEGRATE.md §9).

A single changed pose used to mean six manual stages with stale-data hazards
between each: movegen (changed moves only) → re-apply npz post-edits → bake →
re-trim → certify → prebake. With post edits (BAKE.md §3a), trims and the
loop flag living in the move spec, the whole chain re-runs from one command:

  python pipeline.py --spec moves_runner.json --moves slide
  python pipeline.py --spec moves.json --seeds 8 --groundfix \\
      --char hero.glb --practical ~/src/motionbricks-practical

Stages — each is the plain tool, runnable by hand at any time:
  1. movegen.py     --spec S --only <moves>    (GPU; --skip-gen skips it;
                    auto-wrapped in xvfb-run on headless machines)
  2. bake_moves.py  --spec S                   (CPU FK; re-exports the WHOLE
                    spec so trims / loop flags / frame data are never stale)
  3. certify.mjs    <char> --clips <all baked> (node; only with --char)
  4. prebake.mjs    <char> --manifest ...      (node; only with --char)

The node tools live in the motionbricks-practical checkout: point --practical
(or $MOTIONBRICKS_PRACTICAL) at it. Without --char the chain stops after the
bake and the output is <baked-dir>/ + manifest.json; with it, a failed
certification aborts before prebake — never ship clips through an uncertified
rig. Run from GR00T-WholeBodyControl/motionbricks/ in the MotionBricks env.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def run(stage, cmd):
    print(f"\n[pipeline] {stage}: {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"[pipeline] {stage} failed (exit {r.returncode}) — chain aborted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="moves_example.json")
    ap.add_argument("--moves", default=None,
                    help="comma-separated moves to regenerate (default: all in the spec)")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--groundfix", action="store_true",
                    help="pass --groundfix to movegen (root-lift ground clamp per seed)")
    ap.add_argument("--skip-gen", action="store_true",
                    help="start at the bake (e.g. after editing only trims in the spec)")
    ap.add_argument("--in-dir", default="out/moves")
    ap.add_argument("--baked-dir", default="baked")
    ap.add_argument("--char", default=None,
                    help="character GLB — enables the certify + prebake stages")
    ap.add_argument("--practical", default=os.environ.get("MOTIONBRICKS_PRACTICAL"),
                    help="path to the motionbricks-practical checkout "
                         "(certify.mjs / prebake.mjs); or $MOTIONBRICKS_PRACTICAL")
    ap.add_argument("--prebake-out", default=None,
                    help="prebaked GLB path (default: <char>_anim.glb)")
    ap.add_argument("--rootmotion", default=None,
                    help="rootmotion.json path (default: next to the prebaked GLB)")
    ap.add_argument("--ylift", default=None,
                    help="passed to prebake.mjs, e.g. jump_gap=1.5")
    a = ap.parse_args()

    if not os.path.exists(a.spec):
        sys.exit(f"[pipeline] spec not found: {a.spec}")
    if a.char:
        if not a.practical:
            sys.exit("[pipeline] --char needs --practical (or $MOTIONBRICKS_PRACTICAL) "
                     "to locate certify.mjs / prebake.mjs")
        a.practical = os.path.abspath(os.path.expanduser(a.practical))
        for tool in ("certify.mjs", "prebake.mjs"):
            if not os.path.exists(os.path.join(a.practical, tool)):
                sys.exit(f"[pipeline] {tool} not found in {a.practical}")

    # 1. generate (GPU) — only the changed moves; xvfb on headless machines
    if a.skip_gen:
        print("[pipeline] 1/4 movegen: skipped (--skip-gen)")
    else:
        cmd = [sys.executable, "movegen.py", "--spec", a.spec,
               "--seeds", str(a.seeds), "--out-dir", a.in_dir]
        if a.moves:
            cmd += ["--only", a.moves]
        if a.groundfix:
            cmd.append("--groundfix")
        if not os.environ.get("DISPLAY") and shutil.which("xvfb-run"):
            cmd = ["xvfb-run", "-a"] + cmd
        run("1/4 movegen", cmd)

    # 2. bake (CPU) — the whole spec, so trims/loop/frame data are never stale
    run("2/4 bake", [sys.executable, "bake_moves.py", "--spec", a.spec,
                     "--in-dir", a.in_dir, "--out-dir", a.baked_dir])

    if not a.char:
        print("\n[pipeline] done (no --char: certify/prebake skipped) — "
              f"clips + manifest in {a.baked_dir}/")
        return

    # 3. certify — every baked clip probes the rig; a failure aborts the chain
    with open(os.path.join(a.baked_dir, "manifest.json")) as fp:
        moves = json.load(fp)["moves"]
    clips = ",".join(os.path.join(a.baked_dir, m["file"]) for m in moves)
    run("3/4 certify", ["node", os.path.join(a.practical, "certify.mjs"),
                        a.char, "--clips", clips])

    # 4. prebake — engine-native animations + root motion for the entity layer
    out = a.prebake_out or a.char.replace(".glb", "_anim.glb")
    cmd = ["node", os.path.join(a.practical, "prebake.mjs"), a.char,
           "--manifest", os.path.join(a.baked_dir, "manifest.json"), "--out", out]
    if a.rootmotion:
        cmd += ["--rootmotion", a.rootmotion]
    if a.ylift:
        cmd += ["--ylift", a.ylift]
    run("4/4 prebake", cmd)

    print(f"\n[pipeline] done: {a.baked_dir}/ (clips + manifest), {out}, "
          f"{a.rootmotion or os.path.join(os.path.dirname(out) or '.', 'rootmotion.json')}")


if __name__ == "__main__":
    main()
