# Third-Party Libraries

This project uses the following third-party libraries for the UI:

## Library Versions

### ImGui v1.89.9
- **Repository**: https://github.com/ocornut/imgui
- **Tag**: v1.89.9
- **Commit**: c6e0284a
- **Purpose**: Immediate mode GUI library
- **License**: MIT

### ImPlot v0.16
- **Repository**: https://github.com/epezent/implot
- **Tag**: v0.16
- **Commit**: 18c7243
- **Purpose**: Plotting library for ImGui
- **License**: MIT

### ImGuiFileDialog v0.6.8
- **Repository**: https://github.com/aiekick/ImGuiFileDialog
- **Tag**: v0.6.8
- **Commit**: ee77f19
- **Purpose**: File dialog for ImGui
- **License**: MIT

## Compatibility

These specific versions are tested and compatible with each other:

- **ImGui 1.89.9** works with:
  - ImPlot 0.16 ✓
  - ImGuiFileDialog 0.6.8 ✓

## Setup

### Initial Clone
```bash
git clone --recurse-submodules <repository-url>
```

### Update Submodules
```bash
git submodule update --init --recursive
```

### Check Current Versions
```bash
cd libs/imgui && git describe --tags --always
cd libs/implot && git describe --tags --always  
cd libs/ImGuiFileDialog && git describe --tags --always
```

## API Changes from Previous Versions

### ImGui 1.89.9
- `ListBoxHeader/Footer` → `BeginListBox/EndListBox`
- Backend files moved from `examples/` to `backends/`

### ImPlot 0.16
- `PlotVLines` → `PlotInfLines` (default vertical, use `ImPlotInfLinesFlags_Horizontal` for horizontal)

### ImGuiFileDialog 0.6.8
- `OpenDialog` now requires `IGFD::FileDialogConfig` parameter instead of path string

## Building

These libraries are built as part of the main project via CMake. See `libs/CMakeLists.txt` for build configuration.

## Notes

- All libraries are git repositories but tracked as regular directories
- To convert to proper git submodules, run `./setup_submodules.sh`
- The build system references `imgui/backends/` for ImGui backend implementations
- ImPlot and ImGuiFileDialog depend on ImGui headers being available
