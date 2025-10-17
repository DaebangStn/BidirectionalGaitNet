# Metabolic Energy UI Implementation

## Overview
Added metabolic energy visualization and control features to the GLFW viewer application. Users can now monitor metabolic energy in real-time, switch between different computation modes, and visualize energy consumption through plots.

## Features Implemented

### 1. Metabolic Energy Visualization Panel
**Location**: `viewer/GLFWApp.cpp:1568-1593` (in `drawSimVisualizationPanel()`)

**Components**:
- **Collapsing Header**: "Metabolic Energy" section in the visualization panel
- **Mode Display**: Shows current MetabolicRewardType (LEGACY/A/A2/MA)
- **Current Value Display**: Real-time metabolic energy value with 6 decimal precision
- **Plot**: Time-series graph of metabolic energy using ImPlot

**Code**:
```cpp
// Metabolic Energy
if (ImGui::CollapsingHeader("Metabolic Energy"))
{
    // Display current metabolic reward type
    MetabolicRewardType currentType = mRenderEnv->getCharacter(0)->getMetabolicRewardType();
    const char* typeNames[] = {"LEGACY (Disabled)", "A (abs)", "A2 (squared)", "MA (mass-weighted)"};
    ImGui::Text("Mode: %s", typeNames[currentType]);

    // Display current metabolic energy value
    double metabolicEnergy = mRenderEnv->getCharacter(0)->getMetabolicReward();
    ImGui::Text("Current Value: %.6f", metabolicEnergy);

    ImGui::Separator();

    std::string title_metabolic = mPlotTitle ? mCheckpointName : "Metabolic Energy";
    if (ImPlot::BeginPlot((title_metabolic + "##MetabolicEnergy").c_str()))
    {
        ImPlot::SetupAxes("Time (s)", "Energy");

        // Plot metabolic energy data
        std::vector<std::string> metabolicKeys = {"metabolic_energy"};
        plotGraphData(metabolicKeys, ImAxis_Y1, true, false, "");

        ImPlot::EndPlot();
    }
}
```

### 2. Metabolic Energy Control Panel
**Location**: `viewer/GLFWApp.cpp:2232-2270` (in `drawSimControlPanel()`)

**Components**:
- **Collapsing Header**: "Metabolic Energy" section in the control panel
- **Dropdown Combo Box**: Select MetabolicRewardType mode
  - LEGACY (Disabled)
  - A (abs) - Absolute activation sum
  - A2 (squared) - Squared activation sum
  - MA (mass-weighted) - Mass-weighted absolute activation
- **Current Value Display**: Same real-time value as visualization panel
- **Reset Button**: Manually reset accumulated metabolic energy
- **Help Tooltip**: Hover over "(?)" for mode descriptions and usage notes

**Code**:
```cpp
// Metabolic Energy Control
if (ImGui::CollapsingHeader("Metabolic Energy"))
{
    // Get current metabolic reward type
    MetabolicRewardType currentType = mRenderEnv->getCharacter(0)->getMetabolicRewardType();
    int currentTypeInt = static_cast<int>(currentType);

    // Dropdown for metabolic reward type selection
    const char* metabolicTypes[] = {"LEGACY (Disabled)", "A (abs)", "A2 (squared)", "MA (mass-weighted)"};
    if (ImGui::Combo("Reward Type", &currentTypeInt, metabolicTypes, IM_ARRAYSIZE(metabolicTypes)))
    {
        // Update metabolic reward type when selection changes
        mRenderEnv->getCharacter(0)->setMetabolicRewardType(static_cast<MetabolicRewardType>(currentTypeInt));
    }

    // Display current metabolic energy value
    double metabolicEnergy = mRenderEnv->getCharacter(0)->getMetabolicReward();
    ImGui::Text("Current Value: %.6f", metabolicEnergy);

    // Button to reset metabolic energy accumulation
    if (ImGui::Button("Reset Energy"))
    {
        mRenderEnv->getCharacter(0)->resetMetabolicEnergy();
    }

    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered())
    {
        ImGui::BeginTooltip();
        ImGui::Text("LEGACY: No metabolic computation");
        ImGui::Text("A: Sum of absolute activations");
        ImGui::Text("A2: Sum of squared activations");
        ImGui::Text("MA: Mass-weighted absolute activations");
        ImGui::Separator();
        ImGui::Text("Note: Call cacheMuscleMass() before using MA mode");
        ImGui::EndTooltip();
    }
}
```

## User Workflow

### Basic Usage
1. **Launch the viewer** with environment loaded
2. **Open "Sim Control" panel** (left side)
3. **Expand "Metabolic Energy"** section
4. **Select desired mode** from dropdown:
   - Start with LEGACY to establish baseline
   - Switch to A, A2, or MA to enable energy tracking
5. **Monitor energy** in "Current Value" display
6. **View visualization** in "Sim visualization" panel (right side)
7. **Reset when needed** using "Reset Energy" button

### Advanced Usage with MA Mode
```cpp
// In your setup code (before running simulation)
character->setMuscles(muscle_path);
character->cacheMuscleMass();  // REQUIRED for MA mode
character->setMetabolicRewardType(MA);
```

### Plot Data Integration
The metabolic energy plot uses the existing `plotGraphData()` function, which reads from the graph data buffer. Ensure your simulation/rollout code logs metabolic energy to the key `"metabolic_energy"` in the buffer for the plot to display data.

## UI Design Patterns

### Consistent with Existing UI
- **Collapsing Headers**: Follows same pattern as "Rewards", "Kinematics", "Rollout"
- **ImGui Combo**: Standard dropdown similar to other control selections
- **ImPlot Integration**: Uses same plotting infrastructure as existing graphs
- **Tooltip Pattern**: Help text follows "(?)" convention used throughout

### Real-time Updates
- Values update every frame automatically
- Mode changes take effect immediately
- No need to restart simulation

### Visual Feedback
- Current mode always visible in both panels
- Current value displayed with consistent precision (6 decimals)
- Plot provides historical view of energy consumption

## Technical Implementation Details

### Character API Integration
Uses the public Character interface implemented in `sim/Character.h`:
- `getMetabolicRewardType()` - Query current mode
- `setMetabolicRewardType(type)` - Change computation mode
- `getMetabolicReward()` - Get accumulated average energy
- `resetMetabolicEnergy()` - Clear accumulation

### ImGui/ImPlot Features Used
- `ImGui::CollapsingHeader()` - Expandable sections
- `ImGui::Combo()` - Dropdown selection
- `ImGui::Text()` - Read-only text display
- `ImGui::Button()` - Interactive button
- `ImGui::IsItemHovered()` - Tooltip trigger detection
- `ImGui::BeginTooltip()/EndTooltip()` - Help tooltips
- `ImPlot::BeginPlot()/EndPlot()` - Time-series plotting
- `ImPlot::SetupAxes()` - Axis configuration

### Type Casting
Uses standard C++ casting for enum handling:
```cpp
MetabolicRewardType currentType = ...;
int currentTypeInt = static_cast<int>(currentType);  // Enum to int for ImGui
// ... after combo selection ...
setMetabolicRewardType(static_cast<MetabolicRewardType>(currentTypeInt));  // Int back to enum
```

## Files Modified
- `viewer/GLFWApp.cpp` - Added UI components to both panels
  - Lines 1568-1593: Visualization panel addition
  - Lines 2232-2270: Control panel addition

## Build Verification
✅ Build successful - all 10 targets compiled without errors
✅ No changes required to header files
✅ Uses existing Character API (no new methods needed)
✅ Backward compatible with existing UI

## Integration with Existing Features
- **Plot Title Toggle**: Metabolic plot respects `mPlotTitle` checkbox
- **Checkpoint Names**: Plot title shows checkpoint name when enabled
- **Graph Data Buffer**: Uses existing buffer infrastructure via `plotGraphData()`
- **Character Access**: Accesses character through `mRenderEnv->getCharacter(0)`

## Known Limitations & Notes
1. **MA Mode Requirement**: Must call `cacheMuscleMass()` after `setMuscles()` for MA mode to work correctly
2. **Data Buffer Dependency**: Plot requires "metabolic_energy" key in graph data buffer
3. **Single Character**: Currently only supports first character (index 0)
4. **Mode Persistence**: Mode selection not saved between viewer sessions (defaults to LEGACY)

## Future Enhancements
- Save/load metabolic mode preference in config file
- Multi-character metabolic energy comparison
- Metabolic efficiency metrics (energy per distance traveled)
- Export metabolic energy data to file
- Configurable plot axis ranges
- Color-coded mode indicators (green=efficient, red=wasteful)

## Usage Example

### Interactive Workflow
```
1. Start viewer: ./build/release/viewer/viewer data/trained_nn/merge_no_mesh_lbs
2. Load environment with simulation
3. In Sim Control panel:
   - Expand "Metabolic Energy"
   - Select "A2 (squared)" from dropdown
   - Observe "Current Value" updating
4. In Sim visualization panel:
   - Expand "Metabolic Energy"
   - View real-time plot of energy consumption
5. Run simulation/rollout
6. Compare energy across different modes by switching dropdown
7. Reset accumulation with "Reset Energy" button when needed
```

### Programmatic Setup
```cpp
// Setup in initEnv() or similar initialization
if (mRenderEnv) {
    auto character = mRenderEnv->getCharacter(0);
    character->cacheMuscleMass();  // Enable MA mode support
    character->setMetabolicRewardType(A2);  // Start with A2 mode
}
```

## Troubleshooting

### Plot shows no data
- **Cause**: Graph data buffer doesn't contain "metabolic_energy" key
- **Solution**: Ensure simulation/rollout code logs metabolic energy to buffer

### MA mode shows zero/incorrect values
- **Cause**: `cacheMuscleMass()` not called after `setMuscles()`
- **Solution**: Call `cacheMuscleMass()` in initialization after muscle setup

### Mode changes don't affect plot
- **Cause**: Energy accumulation is averaged over time
- **Solution**: Use "Reset Energy" button to clear previous accumulation

### Dropdown shows incorrect current mode
- **Cause**: Mode was changed externally without UI update
- **Solution**: UI reads current mode every frame, should auto-sync

## References
- Metabolic energy implementation: `claudedocs/metabolic_energy_implementation.md`
- Character API: `sim/Character.h` lines 108-112
- ImGui documentation: https://github.com/ocornut/imgui
- ImPlot documentation: https://github.com/epezent/implot
