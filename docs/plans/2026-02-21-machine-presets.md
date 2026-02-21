# Machine Presets Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire a gitignored `.machine` file to cmake preset selection, add pixi build tasks, replace all shell scripts with pixi tasks, and delete the script files.

**Architecture:** `.machine` holds one preset name (`release`, `debug`, `a6000`, `gait`). Pixi build tasks read it via `$(cat .machine)`. Binary pixi tasks derive the build dir as `build/$(cat .machine)/`. Python pixi tasks call modules directly. All script files in `scripts/` that have pixi equivalents are deleted.

**Tech Stack:** pixi tasks, CMake presets, bash inline tasks

---

### Task 1: Gitignore `.machine` and create it

**Files:**
- Modify: `.gitignore`

**Step 1: Add `.machine` to `.gitignore`**

Append to `/home/geon/BidirectionalGaitNet/.gitignore`:
```
.machine
```

**Step 2: Create `.machine` for this machine**

```bash
printf "release" > /home/geon/BidirectionalGaitNet/.machine
```

**Step 3: Verify**

```bash
cat /home/geon/BidirectionalGaitNet/.machine
git -C /home/geon/BidirectionalGaitNet status .machine
```

Expected: prints `release`; git does NOT show `.machine` (it's ignored).

**Step 4: Commit**

```bash
git -C /home/geon/BidirectionalGaitNet add .gitignore
git -C /home/geon/BidirectionalGaitNet commit -m "chore: gitignore .machine (machine preset selector)"
```

---

### Task 2: Add pixi build tasks

**Files:**
- Modify: `pixi.toml`

**Step 1: Add `[tasks]` section at the end of `pixi.toml`**

Append to `/home/geon/BidirectionalGaitNet/pixi.toml`:

```toml
[tasks]
setup-machine = { cmd = "bash -c 'echo \"Presets: release  debug  a6000  gait\"; printf \"Enter preset name: \"; read p; printf \"%s\" \"$p\" > .machine; echo \"Wrote .machine = $p\"'" }
configure     = "bash -c '[ -f .machine ] || { echo \"Run: pixi run setup-machine\"; exit 1; }; cmake --preset $(cat .machine)'"
build         = "bash -c '[ -f .machine ] || { echo \"Run: pixi run setup-machine\"; exit 1; }; [ -f build/$(cat .machine)/CMakeCache.txt ] || cmake --preset $(cat .machine); cmake --build build/$(cat .machine)/ -j$(nproc) --'"
```

**Step 2: Verify pixi parses the file**

```bash
pixi -C /home/geon/BidirectionalGaitNet task list
```

Expected: `setup-machine`, `configure`, `build` appear in the list.

**Step 3: Smoke-test configure task**

```bash
cd /home/geon/BidirectionalGaitNet && pixi run configure 2>&1 | tail -5
```

Expected: cmake configure completes (or fails only because libs/install/ not yet built — that's OK, we're just checking the task plumbing works, not the full build).

**Step 4: Commit**

```bash
git -C /home/geon/BidirectionalGaitNet add pixi.toml
git -C /home/geon/BidirectionalGaitNet commit -m "build: add pixi configure/build tasks with .machine preset selector"
```

---

### Task 3: Add binary pixi tasks

**Files:**
- Modify: `pixi.toml`

These tasks read `.machine` to resolve the build directory. All set `DISPLAY=:0`. Arguments are passed through using the `bash -c '...' _` pattern (`_` is `$0`, extra pixi args become `$1`, `$2`, ...).

**Step 1: Append binary tasks to the `[tasks]` section in `pixi.toml`**

Add after the `build` task line:

```toml
render-ckpt       = "bash -c 'PRESET=$(cat .machine 2>/dev/null || echo release); CKPT=$1; [ -d \"$CKPT\" ] || CKPT=\"runs/$CKPT\"; shift; export DISPLAY=:0; exec build/${PRESET}/viewer/render_ckpt \"$CKPT\" \"$@\"' _"
surgery           = "bash -c 'PRESET=$(cat .machine 2>/dev/null || echo release); export DISPLAY=:0; exec build/${PRESET}/surgery/surgery_tool \"$@\"' _"
motion-editor     = "bash -c 'PRESET=$(cat .machine 2>/dev/null || echo release); export DISPLAY=:0; exec build/${PRESET}/viewer/motion_editor \"$@\"' _"
marker-editor     = "bash -c 'PRESET=$(cat .machine 2>/dev/null || echo release); export DISPLAY=:0; exec build/${PRESET}/viewer/marker_editor \"$@\"' _"
muscle-personalizer = "bash -c 'PRESET=$(cat .machine 2>/dev/null || echo release); export DISPLAY=:0; exec build/${PRESET}/viewer/muscle_personalizer \"$@\"' _"
physical-exam     = "bash -c 'PRESET=$(cat .machine 2>/dev/null || echo release); export DISPLAY=:0; exec build/${PRESET}/surgery/physical_exam \"$@\"' _"
c3d-inspect       = "bash -c 'PRESET=$(cat .machine 2>/dev/null || echo release); export DISPLAY=:0; exec build/${PRESET}/tools/c3d_inspect \"$@\"' _"
c3d-processor     = "bash -c 'PRESET=$(cat .machine 2>/dev/null || echo release); export DISPLAY=:0; exec build/${PRESET}/viewer/c3d_processor \"$@\"' _"
```

**Step 2: Verify**

```bash
pixi -C /home/geon/BidirectionalGaitNet task list
```

Expected: all 8 binary tasks appear.

**Step 3: Commit**

```bash
git -C /home/geon/BidirectionalGaitNet add pixi.toml
git -C /home/geon/BidirectionalGaitNet commit -m "build: add binary pixi tasks (render-ckpt, surgery, viewers, c3d tools)"
```

---

### Task 4: Add Python pixi tasks

**Files:**
- Modify: `pixi.toml`

Python tasks call modules directly. Arguments are appended by pixi automatically (no `bash -c` wrapper needed since we're not doing shell variable expansion).

**Step 1: Append Python tasks to the `[tasks]` section in `pixi.toml`**

Add after the binary tasks:

```toml
train-bgn      = "python -c 'from python.scripts import train_bgn; train_bgn()'"
train-fgn      = "python -c 'from python.scripts import train_fgn; train_fgn()'"
train-main     = "python -m python.train"
train-pipeline = "python -c 'from python.scripts import train_pipeline; train_pipeline()'"
rollout        = "python -m python.rollout.rollout_cli"
extract-cycle  = "python -c 'from python.scripts import extract_cycle; extract_cycle()'"
```

**Step 2: Verify**

```bash
pixi -C /home/geon/BidirectionalGaitNet task list
```

Expected: all 6 python tasks appear.

**Step 3: Commit**

```bash
git -C /home/geon/BidirectionalGaitNet add pixi.toml
git -C /home/geon/BidirectionalGaitNet commit -m "build: add Python pixi tasks (train, rollout, extract-cycle)"
```

---

### Task 5: Delete replaced script files

**Files:**
- Delete: `scripts/render_ckpt`, `scripts/surgery`, `scripts/motion-editor`, `scripts/marker-editor`, `scripts/muscle-personalizer`, `scripts/physical-exam`, `scripts/c3d_inspect`, `scripts/c3d_processor`, `scripts/train-bgn`, `scripts/train-fgn`, `scripts/train-main`, `scripts/train-pipeline`, `scripts/rollout`, `scripts/extract-cycle`

**Step 1: Delete the script files**

```bash
cd /home/geon/BidirectionalGaitNet
git rm scripts/render_ckpt scripts/surgery scripts/motion-editor scripts/marker-editor \
       scripts/muscle-personalizer scripts/physical-exam scripts/c3d_inspect scripts/c3d_processor \
       scripts/train-bgn scripts/train-fgn scripts/train-main scripts/train-pipeline \
       scripts/rollout scripts/extract-cycle
```

**Step 2: Verify only unrelated scripts remain**

```bash
ls /home/geon/BidirectionalGaitNet/scripts/
```

Expected: only `install.sh`, `launch_ckpts.sh`, `run_cluster.sh`, `plot`, `README.md`, `train/` (cluster scripts) and other utility scripts remain.

**Step 3: Commit**

```bash
git -C /home/geon/BidirectionalGaitNet commit -m "chore: delete scripts replaced by pixi tasks"
```

---

### Task 6: Update `launch_ckpts.sh`

**Files:**
- Modify: `scripts/launch_ckpts.sh`

`launch_ckpts.sh` calls `scripts/render_ckpt` directly and kills processes by the old path. Both references need updating.

**Step 1: Replace the kill line (line ~14)**

Change:
```bash
pkill -f "./build/release/viewer/render_ckpt" 2>/dev/null || true
```
To:
```bash
pkill -f "render_ckpt" 2>/dev/null || true
```

**Step 2: Replace the launch line (~line 53)**

Change:
```bash
        scripts/render_ckpt "${CKPT_PATH}" &
```
To:
```bash
        pixi run render-ckpt -- "${CKPT_PATH}" &
```

**Step 3: Verify the diff**

```bash
git -C /home/geon/BidirectionalGaitNet diff scripts/launch_ckpts.sh
```

Expected: only the two lines above changed.

**Step 4: Commit**

```bash
git -C /home/geon/BidirectionalGaitNet add scripts/launch_ckpts.sh
git -C /home/geon/BidirectionalGaitNet commit -m "fix: update launch_ckpts.sh to use pixi run render-ckpt"
```

---

### Task 7: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Replace the build line**

Change:
```
- You can build the release mode with 'ninja -C build/release -j 16 -l 12'
```
To:
```
- Build with 'pixi run build' (reads preset from .machine; create with 'pixi run setup-machine')
- For direct ninja: 'ninja -C build/$(cat .machine) -j 16 -l 12'
```

**Step 2: Commit**

```bash
git -C /home/geon/BidirectionalGaitNet add CLAUDE.md
git -C /home/geon/BidirectionalGaitNet commit -m "docs: update CLAUDE.md build command to pixi run build"
```
