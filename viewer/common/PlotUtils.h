#ifndef PLOT_UTILS_H
#define PLOT_UTILS_H

#include <implot.h>
#include <imgui.h>
#include <vector>
#include <string>
#include <map>
#include "CBufferData.h"

namespace PlotUtils
{
    /**
     * @brief Plot multiple time-series data with optional statistics
     * @param graphData CBufferData containing value buffers
     * @param keys Keys to plot
     * @param y_axis Which Y axis to use (ImAxis_Y1, ImAxis_Y2, etc.)
     * @param postfix Label postfix
     * @param show_stat Show statistics in legend
     * @param color_ofs Color offset for cycling through colors
     */
    void plotGraphData(
        CBufferData<double>* graphData,
        const std::vector<std::string>& keys,
        ImAxis y_axis = ImAxis_Y1,
        const std::string& postfix = "",
        bool show_stat = false,
        int color_ofs = 0
    );

    /**
     * @brief Compute statistics for graph data within an index range
     * @param graphData CBufferData containing value buffers
     * @param keys Keys to compute stats for
     * @param xMin Minimum index
     * @param xMax Maximum index
     * @return Map of key -> {mean, std, min, max}
     */
    std::map<std::string, std::map<std::string, double>> statGraphData(
        CBufferData<double>* graphData,
        const std::vector<std::string>& keys,
        double xMin,
        double xMax
    );

    /**
     * @brief Plot marker error over time
     * @param graphData CBufferData containing value buffers
     * @param xMin X-axis minimum
     * @param plotHeight Plot height in pixels
     */
    void plotMarkerError(
        CBufferData<double>* graphData,
        double xMin,
        float plotHeight = 200.0f
    );

} // namespace PlotUtils

#endif // PLOT_UTILS_H
