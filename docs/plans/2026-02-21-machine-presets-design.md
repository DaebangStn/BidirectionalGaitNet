# Machine Presets Design

Date: 2026-02-21

## Goal

Replace hardcoded `build/release/` paths with a `.machine` file that selects the cmake preset per machine. Add pixi tasks for configure/build. Replace all shell scripts with pixi tasks.

## Section 1: `.machine` file

- Add `.machine` to `.gitignore` (non-tracked, machine-local)
- File contains one line: the cmake preset name (`release`, `debug`, `a6000`, `gait`)
- Create with `release` on this machine
- Scripts/tasks fall back to `release` if absent

## Section 2: Pixi build tasks

```toml
[tasks]
setup-machine = interactive prompt that writes preset name to .machine
configure     = cmake --preset $(cat .machine)
build         = configure if needed + cmake --build build/$(cat .machine)/ -j$(nproc)
```

## Section 3: Replace shell scripts with pixi tasks

Binary scripts (require DISPLAY=:0, resolve build dir from .machine):
- render-ckpt  → build/$(cat .machine)/viewer/render_ckpt  (preserves runs/ fallback logic)
- surgery       → build/$(cat .machine)/surgery/surgery_tool
- motion-editor → build/$(cat .machine)/viewer/motion_editor
- marker-editor → build/$(cat .machine)/viewer/marker_editor
- muscle-personalizer → build/$(cat .machine)/viewer/muscle_personalizer
- physical-exam → build/$(cat .machine)/surgery/physical_exam

Python tasks (no build dir, pure python):
- train-bgn     → python.scripts.train_bgn
- train-fgn     → python.scripts.train_fgn
- train-main    → python.train module
- train-pipeline → python.scripts.train_pipeline
- rollout       → python.rollout.rollout_cli

All script files deleted after tasks are added.

## Section 4: CLAUDE.md

Update build command from `ninja -C build/release -j 16 -l 12` to `pixi run build`.
