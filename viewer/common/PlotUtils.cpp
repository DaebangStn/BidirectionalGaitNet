#include "PlotUtils.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace PlotUtils
{

void plotGraphData(
    CBufferData<double>* graphData,
    const std::vector<std::string>& keys,
    ImAxis y_axis,
    const std::string& postfix,
    bool show_stat,
    int color_ofs)
{
    if (keys.empty() || !graphData) return;

    ImPlot::SetAxis(y_axis);

    // Compute statistics for current plot range if show_stat is enabled
    std::map<std::string, std::map<std::string, double>> stats;
    if (show_stat) {
        ImPlotRect limits = ImPlot::GetPlotLimits();
        stats = statGraphData(graphData, keys, limits.X.Min, limits.X.Max);
    }

    // Get colormap size for stable color assignment
    int colormapSize = ImPlot::GetColormapSize();
    int keyIndex = 0;

    for (const auto& key : keys) {
        if (!graphData->key_exists(key)) {
            std::cerr << "Key " << key << " not found in graph data" << std::endl;
            continue;
        }

        std::vector<double> values = graphData->get(key);
        if (values.empty()) {
            continue;
        }

        int bufferSize = static_cast<int>(values.size());

        // Create x-axis data (index-based: most recent at 0, oldest at -N)
        std::vector<float> x(bufferSize);
        std::vector<float> y(bufferSize);
        for (int i = 0; i < bufferSize; ++i) {
            x[i] = static_cast<float>(-(bufferSize - 1 - i));
            y[i] = static_cast<float>(values[i]);
        }

        // Format key name (strip prefix before first underscore)
        std::string selected_key = key;
        size_t underscore_pos = key.find('_');
        if (underscore_pos != std::string::npos) {
            selected_key = key.substr(underscore_pos + 1);
        }
        selected_key = selected_key + postfix;

        // Build plot label with stable ID to prevent color flickering
        std::string plot_label = selected_key;

        // Append statistics to legend if enabled
        if (show_stat && stats.count(key) > 0) {
            char stat_str[128];
            snprintf(stat_str, sizeof(stat_str), " (%.2f|%.2f|%.2f)",
                     stats[key]["min"], stats[key]["mean"], stats[key]["max"]);
            plot_label += stat_str;
        }

        // Add stable ID to prevent color changes when stats update
        plot_label += "##" + key;

        // Assign stable color based on key index
        int colorIndex = (keyIndex + color_ofs) % colormapSize;
        if (colorIndex < 0) colorIndex += colormapSize;
        ImPlot::SetNextLineStyle(ImPlot::GetColormapColor(colorIndex));

        // Plot the data
        ImPlot::PlotLine(plot_label.c_str(), x.data(), y.data(), bufferSize);

        keyIndex++;
    }
}

std::map<std::string, std::map<std::string, double>> statGraphData(
    CBufferData<double>* graphData,
    const std::vector<std::string>& keys,
    double xMin,
    double xMax)
{
    std::map<std::string, std::map<std::string, double>> result;

    if (!graphData) return result;

    for (const auto& key : keys) {
        if (!graphData->key_exists(key)) continue;

        std::vector<double> values = graphData->get(key);
        if (values.empty()) continue;

        int bufferSize = static_cast<int>(values.size());

        // Find values within index range
        // X-axis mapping: most recent data at x=0, older data at negative x values
        std::vector<double> rangeValues;
        for (int i = 0; i < bufferSize; ++i) {
            double idx = -(bufferSize - 1 - i);
            if (idx >= xMin && idx <= xMax) {
                rangeValues.push_back(values[i]);
            }
        }

        if (rangeValues.empty()) {
            continue;
        }

        // Compute statistics
        double sum = std::accumulate(rangeValues.begin(), rangeValues.end(), 0.0);
        double mean = sum / rangeValues.size();

        double sq_sum = std::inner_product(rangeValues.begin(), rangeValues.end(),
                                          rangeValues.begin(), 0.0);
        double variance = (sq_sum / rangeValues.size()) - (mean * mean);
        double std = std::sqrt(std::max(0.0, variance));

        double min_val = *std::min_element(rangeValues.begin(), rangeValues.end());
        double max_val = *std::max_element(rangeValues.begin(), rangeValues.end());

        result[key]["mean"] = mean;
        result[key]["std"] = std;
        result[key]["min"] = min_val;
        result[key]["max"] = max_val;
    }

    return result;
}

void plotMarkerError(
    CBufferData<double>* graphData,
    double xMin,
    float plotHeight)
{
    static bool showStat = false;
    ImGui::Checkbox("Show Statistics", &showStat);

    // Check if data exists
    if (!graphData || !graphData->key_exists("marker_error_mean")) {
        return;
    }

    ImPlot::SetNextAxisLimits(ImAxis_X1, -100, 0.0, ImGuiCond_Always);
    ImPlot::SetNextAxisLimits(ImAxis_Y1, 0.0, 100.0, ImGuiCond_Once);

    if (ImPlot::BeginPlot("Marker Error##MarkerDiff", ImVec2(-1, plotHeight))) {
        ImPlot::SetupAxes("Frame", "Error (mm)");

        std::vector<std::string> keys = {"marker_error_mean"};
        plotGraphData(graphData, keys, ImAxis_Y1, "", showStat, 0);

        ImPlot::EndPlot();
    }
}

} // namespace PlotUtils
