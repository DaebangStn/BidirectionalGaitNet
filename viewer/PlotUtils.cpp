#include "PlotUtils.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace PlotUtils
{

void plotGraphData(
    const std::map<std::string, std::vector<double>>& graphData,
    const std::map<std::string, std::vector<double>>& graphTime,
    const std::vector<std::string>& keys,
    ImAxis y_axis,
    const std::string& postfix,
    bool show_stat,
    int color_ofs,
    double timeStep)
{
    if (keys.empty()) return;

    ImPlot::SetAxis(y_axis);

    // Compute statistics for current plot range if show_stat is enabled
    std::map<std::string, std::map<std::string, double>> stats;
    if (show_stat) {
        ImPlotRect limits = ImPlot::GetPlotLimits();
        stats = statGraphData(graphData, graphTime, keys, limits.X.Min, limits.X.Max);
    }

    // Get colormap size for stable color assignment
    int colormapSize = ImPlot::GetColormapSize();
    int keyIndex = 0;

    for (const auto& key : keys) {
        auto dataIt = graphData.find(key);
        auto timeIt = graphTime.find(key);

        if (dataIt == graphData.end() || timeIt == graphTime.end()) {
            std::cerr << "Key " << key << " not found in graph data" << std::endl;
            continue;
        }

        const auto& values = dataIt->second;
        const auto& times = timeIt->second;

        if (values.empty() || times.empty()) {
            continue;
        }

        // Ensure time and value vectors have same size
        size_t minSize = std::min(times.size(), values.size());

        // Build label with optional statistics
        std::string label = key + postfix;
        if (show_stat && stats.find(key) != stats.end()) {
            const auto& keyStat = stats[key];
            char buf[256];
            snprintf(buf, sizeof(buf), "%s (μ=%.3f σ=%.3f)",
                     label.c_str(),
                     keyStat.at("mean"),
                     keyStat.at("std"));
            label = buf;
        }

        // Set plot color based on key index with offset
        int colorIdx = (keyIndex + color_ofs) % colormapSize;
        ImPlot::SetNextLineStyle(ImPlot::GetColormapColor(colorIdx));

        // Plot the data
        ImPlot::PlotLine(label.c_str(),
                        times.data(),
                        values.data(),
                        static_cast<int>(minSize));

        keyIndex++;
    }
}

std::map<std::string, std::map<std::string, double>> statGraphData(
    const std::map<std::string, std::vector<double>>& graphData,
    const std::map<std::string, std::vector<double>>& graphTime,
    const std::vector<std::string>& keys,
    double xMin,
    double xMax)
{
    std::map<std::string, std::map<std::string, double>> result;

    for (const auto& key : keys) {
        auto dataIt = graphData.find(key);
        auto timeIt = graphTime.find(key);

        if (dataIt == graphData.end() || timeIt == graphTime.end()) {
            continue;
        }

        const auto& values = dataIt->second;
        const auto& times = timeIt->second;

        if (values.empty() || times.empty()) {
            continue;
        }

        // Find values within time range
        std::vector<double> rangeValues;
        size_t minSize = std::min(times.size(), values.size());

        for (size_t i = 0; i < minSize; ++i) {
            if (times[i] >= xMin && times[i] <= xMax) {
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
    const std::map<std::string, std::vector<double>>& graphData,
    const std::map<std::string, std::vector<double>>& graphTime,
    double xMin,
    float plotHeight)
{
    auto dataIt = graphData.find("marker_error_mean");
    auto timeIt = graphTime.find("marker_error_mean");

    if (dataIt == graphData.end() || timeIt == graphTime.end()) {
        return;
    }

    const auto& times = timeIt->second;
    const auto& values = dataIt->second;

    if (times.empty() || values.empty()) {
        return;
    }

    ImPlot::SetNextAxisLimits(ImAxis_X1, xMin, 0, ImGuiCond_Always);
    ImPlot::SetNextAxisLimits(ImAxis_Y1, 0.0, 0.3, ImGuiCond_Once);

    if (ImPlot::BeginPlot("Marker Error##MarkerDiff", ImVec2(-1, plotHeight))) {
        ImPlot::SetupAxes("Time (s)", "Error (m)");

        // Show last 500 points
        size_t numPoints = std::min(times.size(), size_t(500));
        std::vector<double> t(times.end() - numPoints, times.end());
        std::vector<double> v(values.end() - numPoints, values.end());

        if (!t.empty()) {
            ImPlot::PlotLine("Mean Error", t.data(), v.data(), static_cast<int>(t.size()));
        }

        ImPlot::EndPlot();
    }
}

} // namespace PlotUtils
