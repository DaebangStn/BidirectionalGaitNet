#include <H5Cpp.h>
#include <ncurses.h>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <getopt.h>

namespace fs = std::filesystem;

// Data structures
struct CycleInfo {
    int index;
    int frames;
    double duration;
    double phase_min;
    double phase_max;
    bool is_param_averaged;  // True if this represents param-level averaged data

    CycleInfo() : index(-1), frames(0), duration(0.0), phase_min(0.0), phase_max(0.0), is_param_averaged(false) {}
};

struct ParamInfo {
    int index;
    int timesteps;      // Limited to 5000+
    int cycle_count;     // Limited to 100+
    std::vector<CycleInfo> cycles;
};

struct FileInfo {
    std::string path;
    std::string name;
    int timesteps;      // Limited to 5000+
    int param_count;     // Limited to 100+
    int cycle_count;     // Limited to 100+
    std::vector<ParamInfo> params;
};

// Constants
const int MAX_TIMESTEPS_COUNT = 10000;
const int MAX_PARAM_COUNT = 100;
const int MAX_CYCLE_COUNT = 100;

// UI State
enum Stage { FILE_SELECT, PARAM_SELECT, CYCLE_SELECT, CONFIRM };

class ExtractorUI {
public:
    ExtractorUI() : current_stage(FILE_SELECT), selected_file_idx(0),
                    selected_param_idx(0), selected_cycle_idx(0) {}

    void run();
    bool extractDirect(const std::string& filepath, int param_idx, int cycle_idx,
                      const std::string& output_path);

private:
    Stage current_stage;
    int selected_file_idx;
    int selected_param_idx;
    int selected_cycle_idx;

    std::vector<FileInfo> files;

    void scanFiles();
    void loadFileInfo(FileInfo& file_info);
    void loadParamInfo(const std::string& filepath, ParamInfo& param_info);
    void loadCycleInfo(const std::string& filepath, int param_idx, CycleInfo& cycle_info);

    void stage1_fileSelect();
    void stage2_paramSelect();
    void stage3_cycleSelect();
    void stageConfirm();

    void extractCycle(const std::string& input_path, int param_idx, int cycle_idx,
                     const std::string& output_path);

    void drawUI();
    void drawFileList();
    void drawParamList();
    void drawCycleList();
    void drawConfirmation();
};

void ExtractorUI::scanFiles() {
    files.clear();

    std::string sampled_dir = "sampled";
    if (!fs::exists(sampled_dir)) {
        printw("Error: sampled/ directory not found\n");
        return;
    }

    for (const auto& entry : fs::directory_iterator(sampled_dir)) {
        if (entry.is_directory()) {
            for (const auto& file : fs::directory_iterator(entry.path())) {
                if (file.path().extension() == ".h5" || file.path().extension() == ".hdf5") {
                    FileInfo info;
                    info.path = file.path().string();
                    info.name = file.path().filename().string();
                    info.timesteps = 0;
                    info.param_count = 0;
                    info.cycle_count = 0;
                    files.push_back(info);
                }
            }
        }
    }

    // Load info for each file
    for (auto& file : files) {
        loadFileInfo(file);
    }
}

void ExtractorUI::loadFileInfo(FileInfo& file_info) {
    try {
        H5::H5File file(file_info.path, H5F_ACC_RDONLY);

        int total_timesteps = 0;
        int param_count = 0;
        int total_cycles = 0;

        // Scan for param_N groups - count ALL params
        for (int p = 0; p < MAX_PARAM_COUNT; p++) {
            std::stringstream ss;
            ss << "param_" << p;
            std::string param_name = ss.str();

            if (!H5Lexists(file.getId(), param_name.c_str(), H5P_DEFAULT)) {
                break;
            }

            param_count++;

            // Count all cycles and timesteps (no limit on discovery)
            int param_timesteps = 0;
            for (int c = 0; c < MAX_CYCLE_COUNT; c++) {
                std::stringstream ss_cycle;
                ss_cycle << param_name << "/cycle_" << c;
                std::string cycle_path = ss_cycle.str();

                if (!H5Lexists(file.getId(), cycle_path.c_str(), H5P_DEFAULT)) {
                    break;
                }

                total_cycles++;

                // Get frame count
                std::string motions_path = cycle_path + "/motions";
                H5::DataSet dataset = file.openDataSet(motions_path);
                H5::DataSpace dataspace = dataset.getSpace();
                hsize_t dims[2];
                dataspace.getSimpleExtentDims(dims, nullptr);

                param_timesteps += static_cast<int>(dims[0]);
                total_timesteps += static_cast<int>(dims[0]);
            }

            if (param_count >= MAX_PARAM_COUNT) {
                break;
            }
        }

        file_info.timesteps = total_timesteps;
        file_info.param_count = param_count;
        file_info.cycle_count = total_cycles;

        file.close();
    } catch (const H5::Exception& e) {
        file_info.timesteps = -1;
        file_info.param_count = -1;
        file_info.cycle_count = -1;
    }
}

void ExtractorUI::loadParamInfo(const std::string& filepath, ParamInfo& param_info) {
    try {
        H5::H5File file(filepath, H5F_ACC_RDONLY);

        int total_timesteps = 0;
        int cycle_count = 0;

        // Load all cycles without timestep limit (limit only applies to extraction)
        for (int c = 0; c < MAX_CYCLE_COUNT; c++) {
            std::stringstream ss;
            ss << "param_" << param_info.index << "/cycle_" << c;
            std::string cycle_path = ss.str();

            if (!H5Lexists(file.getId(), cycle_path.c_str(), H5P_DEFAULT)) {
                break;
            }

            cycle_count++;

            std::string motions_path = cycle_path + "/motions";
            H5::DataSet dataset = file.openDataSet(motions_path);
            H5::DataSpace dataspace = dataset.getSpace();
            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims, nullptr);

            total_timesteps += static_cast<int>(dims[0]);
        }

        param_info.timesteps = total_timesteps;
        param_info.cycle_count = cycle_count;

        file.close();
    } catch (const H5::Exception& e) {
        param_info.timesteps = -1;
        param_info.cycle_count = -1;
    }
}

void ExtractorUI::loadCycleInfo(const std::string& filepath, int param_idx, CycleInfo& cycle_info) {
    try {
        H5::H5File file(filepath, H5F_ACC_RDONLY);

        std::stringstream ss;
        ss << "param_" << param_idx << "/cycle_" << cycle_info.index;
        std::string base_path = ss.str();

        // Load motions dimensions
        std::string motions_path = base_path + "/motions";
        H5::DataSet dataset_motions = file.openDataSet(motions_path);
        H5::DataSpace dataspace_motions = dataset_motions.getSpace();
        hsize_t dims[2];
        dataspace_motions.getSimpleExtentDims(dims, nullptr);
        cycle_info.frames = static_cast<int>(dims[0]);

        // Load phase data
        std::string phase_path = base_path + "/phase";
        H5::DataSet dataset_phase = file.openDataSet(phase_path);
        H5::DataSpace dataspace_phase = dataset_phase.getSpace();
        hsize_t phase_dims[1];
        dataspace_phase.getSimpleExtentDims(phase_dims, nullptr);

        std::vector<float> phase_data(phase_dims[0]);
        dataset_phase.read(phase_data.data(), H5::PredType::NATIVE_FLOAT);

        cycle_info.phase_min = *std::min_element(phase_data.begin(), phase_data.end());
        cycle_info.phase_max = *std::max_element(phase_data.begin(), phase_data.end());

        // Load time data
        std::string time_path = base_path + "/time";
        H5::DataSet dataset_time = file.openDataSet(time_path);
        H5::DataSpace dataspace_time = dataset_time.getSpace();
        hsize_t time_dims[1];
        dataspace_time.getSimpleExtentDims(time_dims, nullptr);

        std::vector<float> time_data(time_dims[0]);
        dataset_time.read(time_data.data(), H5::PredType::NATIVE_FLOAT);

        cycle_info.duration = time_data.back() - time_data.front();

        file.close();
    } catch (const H5::Exception& e) {
        cycle_info.frames = -1;
        cycle_info.duration = -1.0;
        cycle_info.phase_min = 0.0;
        cycle_info.phase_max = 0.0;
    }
}

void ExtractorUI::stage1_fileSelect() {
    clear();
    mvprintw(0, 0, "=== HDF5 Cycle Extractor - Stage 1/3: File Selection ===\n\n");

    drawFileList();

    mvprintw(LINES - 2, 0, "[↑/↓: Navigate] [Enter: Select] [q: Quit]");
    refresh();

    int ch = getch();
    if (ch == 'q' || ch == 'Q') {
        endwin();
        exit(0);
    } else if (ch == KEY_UP && selected_file_idx > 0) {
        selected_file_idx--;
    } else if (ch == KEY_DOWN && selected_file_idx < static_cast<int>(files.size()) - 1) {
        selected_file_idx++;
    } else if (ch == '\n' || ch == KEY_ENTER || ch == 10) {
        // Load params for selected file
        FileInfo& file = files[selected_file_idx];
        file.params.clear();

        try {
            H5::H5File h5file(file.path, H5F_ACC_RDONLY);
            for (int p = 0; p < file.param_count; p++) {
                ParamInfo param;
                param.index = p;
                loadParamInfo(file.path, param);
                file.params.push_back(param);
            }
            h5file.close();
        } catch (...) {}

        // Auto-skip if only one file
        if (files.size() == 1) {
            mvprintw(LINES - 4, 0, "Only one file found, auto-selecting...");
            refresh();
            napms(1000);
        }

        // Auto-skip if only one param
        if (file.params.size() == 1) {
            selected_param_idx = 0;
            ParamInfo& param = file.params[selected_param_idx];

            // Load cycles
            param.cycles.clear();
            for (int c = 0; c < param.cycle_count; c++) {
                CycleInfo cycle;
                cycle.index = c;
                loadCycleInfo(file.path, param.index, cycle);
                param.cycles.push_back(cycle);
            }

            // Auto-skip if only one cycle
            if (param.cycles.size() == 1) {
                selected_cycle_idx = 0;
                current_stage = CONFIRM;
            } else {
                current_stage = CYCLE_SELECT;
            }
        } else {
            current_stage = PARAM_SELECT;
        }
    }
}

void ExtractorUI::stage2_paramSelect() {
    clear();
    mvprintw(0, 0, "=== HDF5 Cycle Extractor - Stage 2/3: Param Selection ===\n\n");

    drawParamList();

    mvprintw(LINES - 2, 0, "[↑/↓: Navigate] [Enter: Select] [b: Back] [q: Quit]");
    refresh();

    int ch = getch();
    if (ch == 'q' || ch == 'Q') {
        endwin();
        exit(0);
    } else if (ch == 'b' || ch == 'B') {
        current_stage = FILE_SELECT;
    } else if (ch == KEY_UP && selected_param_idx > 0) {
        selected_param_idx--;
    } else if (ch == KEY_DOWN && selected_param_idx < static_cast<int>(files[selected_file_idx].params.size()) - 1) {
        selected_param_idx++;
    } else if (ch == '\n' || ch == KEY_ENTER || ch == 10) {
        ParamInfo& param = files[selected_file_idx].params[selected_param_idx];

        // Load cycles
        param.cycles.clear();
        for (int c = 0; c < param.cycle_count; c++) {
            CycleInfo cycle;
            cycle.index = c;
            loadCycleInfo(files[selected_file_idx].path, param.index, cycle);
            param.cycles.push_back(cycle);
        }

        // Auto-skip if only one cycle
        if (param.cycles.size() == 1) {
            selected_cycle_idx = 0;
            current_stage = CONFIRM;
        } else {
            current_stage = CYCLE_SELECT;
        }
    }
}

void ExtractorUI::stage3_cycleSelect() {
    clear();
    mvprintw(0, 0, "=== HDF5 Cycle Extractor - Stage 3/3: Cycle Selection ===\n\n");

    drawCycleList();

    mvprintw(LINES - 2, 0, "[↑/↓: Navigate] [Enter: Select] [b: Back] [q: Quit]");
    refresh();

    // Check if param-level data exists
    bool has_param_data = false;
    try {
        H5::H5File h5file(files[selected_file_idx].path, H5F_ACC_RDONLY);
        std::stringstream ss;
        ss << "param_" << files[selected_file_idx].params[selected_param_idx].index << "/motions";
        has_param_data = H5Lexists(h5file.getId(), ss.str().c_str(), H5P_DEFAULT);
        h5file.close();
    } catch (...) {}

    int ch = getch();
    if (ch == 'q' || ch == 'Q') {
        endwin();
        exit(0);
    } else if (ch == 'b' || ch == 'B') {
        current_stage = PARAM_SELECT;
    } else if (ch == KEY_UP) {
        // Allow going up to -1 (param_averaged) if param data exists
        int min_idx = has_param_data ? -1 : 0;
        if (selected_cycle_idx > min_idx) {
            selected_cycle_idx--;
        }
    } else if (ch == KEY_DOWN) {
        ParamInfo& param = files[selected_file_idx].params[selected_param_idx];
        if (selected_cycle_idx < static_cast<int>(param.cycles.size()) - 1) {
            selected_cycle_idx++;
        }
    } else if (ch == '\n' || ch == KEY_ENTER || ch == 10) {
        current_stage = CONFIRM;
    }
}

void ExtractorUI::stageConfirm() {
    clear();
    mvprintw(0, 0, "=== HDF5 Cycle Extractor - Confirmation ===\n\n");

    drawConfirmation();

    mvprintw(LINES - 2, 0, "[y: Extract] [b: Back] [q: Quit]");
    refresh();

    int ch = getch();
    if (ch == 'q' || ch == 'Q') {
        endwin();
        exit(0);
    } else if (ch == 'b' || ch == 'B') {
        current_stage = CYCLE_SELECT;
    } else if (ch == 'y' || ch == 'Y') {
        // Generate output filename
        FileInfo& file = files[selected_file_idx];
        ParamInfo& param = file.params[selected_param_idx];
        bool is_param_averaged = (selected_cycle_idx == -1);

        // Ensure data/motion directory exists
        fs::create_directories("data/motion");

        std::string base_name = file.name.substr(0, file.name.find_last_of('.'));
        std::stringstream ss;
        ss << "data/motion/" << base_name << "_param" << param.index;
        if (is_param_averaged) {
            ss << "_averaged.h5";
        } else {
            ss << "_cycle" << param.cycles[selected_cycle_idx].index << ".h5";
        }
        std::string output_path = ss.str();

        fs::path rel_output = fs::relative(output_path);
        mvprintw(LINES - 4, 0, "Extracting to: %s", rel_output.string().c_str());
        refresh();

        extractCycle(file.path, param.index, selected_cycle_idx, output_path);

        mvprintw(LINES - 3, 0, "Extraction complete! Press any key to exit...");
        refresh();
        getch();
        endwin();
        exit(0);
    }
}

void ExtractorUI::drawFileList() {
    printw("Files found in sampled/:\n\n");

    for (size_t i = 0; i < files.size(); i++) {
        if (i == selected_file_idx) printw("  > ");
        else printw("    ");

        fs::path rel_path = fs::relative(files[i].path);
        printw("%s\n", rel_path.string().c_str());
        printw("      Timesteps: %s  Params: %s  Cycles: %s\n",
               files[i].timesteps >= MAX_TIMESTEPS_COUNT ? "5000+" : std::to_string(files[i].timesteps).c_str(),
               files[i].param_count >= MAX_PARAM_COUNT ? "100+" : std::to_string(files[i].param_count).c_str(),
               files[i].cycle_count >= MAX_CYCLE_COUNT ? "100+" : std::to_string(files[i].cycle_count).c_str());
        printw("\n");
    }
}

void ExtractorUI::drawParamList() {
    FileInfo& file = files[selected_file_idx];
    fs::path rel_path = fs::relative(file.path);
    printw("File: %s\n\n", rel_path.string().c_str());
    printw("Params:\n\n");

    for (size_t i = 0; i < file.params.size(); i++) {
        if (i == selected_param_idx) printw("  > ");
        else printw("    ");

        ParamInfo& param = file.params[i];
        printw("param_%d:  Timesteps: %s  Cycles: %d\n",
               param.index,
               param.timesteps >= MAX_TIMESTEPS_COUNT ? "5000+" : std::to_string(param.timesteps).c_str(),
               param.cycle_count);
    }
}

void ExtractorUI::drawCycleList() {
    FileInfo& file = files[selected_file_idx];
    ParamInfo& param = file.params[selected_param_idx];
    fs::path rel_path = fs::relative(file.path);
    printw("File: %s\n", rel_path.string().c_str());
    printw("Param: %d\n\n", param.index);

    // Check for param-level averaged data and get info
    bool has_param_data = false;
    int param_samples = 0;
    double param_duration = 0.0;
    double param_phase_min = 0.0;
    double param_phase_max = 1.0;

    try {
        H5::H5File h5file(file.path, H5F_ACC_RDONLY);
        std::stringstream ss_base;
        ss_base << "param_" << param.index;
        std::string param_path = ss_base.str();

        // Check motions
        std::string motions_path = param_path + "/motions";
        if (H5Lexists(h5file.getId(), motions_path.c_str(), H5P_DEFAULT)) {
            has_param_data = true;
            H5::DataSet dataset = h5file.openDataSet(motions_path);
            H5::DataSpace dataspace = dataset.getSpace();
            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims, nullptr);
            param_samples = static_cast<int>(dims[0]);
        }

        // Get time range if available
        std::string time_path = param_path + "/time";
        if (H5Lexists(h5file.getId(), time_path.c_str(), H5P_DEFAULT)) {
            H5::DataSet time_dataset = h5file.openDataSet(time_path);
            std::vector<float> time_data(param_samples);
            time_dataset.read(time_data.data(), H5::PredType::NATIVE_FLOAT);
            param_duration = time_data.back() - time_data.front();
        }

        // Get phase range if available
        std::string phase_path = param_path + "/phase";
        if (H5Lexists(h5file.getId(), phase_path.c_str(), H5P_DEFAULT)) {
            H5::DataSet phase_dataset = h5file.openDataSet(phase_path);
            std::vector<float> phase_data(param_samples);
            phase_dataset.read(phase_data.data(), H5::PredType::NATIVE_FLOAT);
            param_phase_min = *std::min_element(phase_data.begin(), phase_data.end());
            param_phase_max = *std::max_element(phase_data.begin(), phase_data.end());
        }

        h5file.close();
    } catch (...) {}

    printw("Cycles:\n\n");

    // Calculate available space for cycle list
    int available_lines = LINES - 8;  // Header takes 5 lines, footer needs 3 lines
    int total_items = param.cycles.size() + (has_param_data ? 1 : 0);

    // Calculate scroll window
    int scroll_offset = 0;
    if (total_items > available_lines) {
        // Center selected item in visible window
        scroll_offset = selected_cycle_idx - (available_lines / 2);
        if (scroll_offset < 0) scroll_offset = 0;
        if (scroll_offset + available_lines > total_items) {
            scroll_offset = total_items - available_lines;
        }
    }

    int visible_end = std::min(scroll_offset + available_lines, total_items);

    // Show scroll indicator if needed
    if (scroll_offset > 0) {
        printw("    ... (%d more items above) ...\n", scroll_offset);
    }

    // Show param-level averaged data first (if exists and in visible range)
    if (has_param_data && scroll_offset == 0) {
        if (selected_cycle_idx == -1) printw("  > ");
        else printw("    ");
        printw("param_averaged:  Frames: %d  Duration: %.2f s  Phase: [%.3f, %.3f]\n",
               param_samples, param_duration, param_phase_min, param_phase_max);
    }

    // Show visible cycles (adjust indices based on param_data offset)
    int cycle_start = has_param_data ? std::max(0, scroll_offset - 1) : scroll_offset;
    int cycle_end = has_param_data ? std::min((int)param.cycles.size(), visible_end - 1) : visible_end;

    for (int i = cycle_start; i < cycle_end; i++) {
        int display_idx = i + (has_param_data ? 1 : 0);
        if (display_idx < scroll_offset || display_idx >= visible_end) continue;

        if (i == selected_cycle_idx) printw("  > ");
        else printw("    ");

        CycleInfo& cycle = param.cycles[i];
        printw("cycle_%d:  Frames: %d  Duration: %.2f s  Phase: [%.3f, %.3f]\n",
               cycle.index, cycle.frames, cycle.duration,
               cycle.phase_min, cycle.phase_max);
    }

    // Show scroll indicator if needed
    if (visible_end < total_items) {
        printw("    ... (%d more items below) ...\n", total_items - visible_end);
    }
}

void ExtractorUI::drawConfirmation() {
    FileInfo& file = files[selected_file_idx];
    ParamInfo& param = file.params[selected_param_idx];
    bool is_param_averaged = (selected_cycle_idx == -1);

    // Get relative paths
    fs::path rel_input = fs::relative(file.path);

    std::string base_name = file.name.substr(0, file.name.find_last_of('.'));
    std::stringstream ss_output;
    ss_output << "data/motion/" << base_name << "_param" << param.index;
    if (is_param_averaged) {
        ss_output << "_averaged.h5";
    } else {
        ss_output << "_cycle" << param.cycles[selected_cycle_idx].index << ".h5";
    }
    fs::path rel_output = fs::relative(ss_output.str());

    printw("Selected:\n");
    printw("---------\n");
    printw("File:  %s\n", rel_input.string().c_str());
    printw("Param: %d\n", param.index);
    if (is_param_averaged) {
        printw("Type:  param_averaged\n\n");
    } else {
        printw("Cycle: %d\n\n", param.cycles[selected_cycle_idx].index);
    }

    if (is_param_averaged) {
        // Get param-level data info
        int param_samples = 0;
        double param_duration = 0.0;
        double param_phase_min = 0.0;
        double param_phase_max = 1.0;

        try {
            H5::H5File h5file(file.path, H5F_ACC_RDONLY);
            std::stringstream ss_base;
            ss_base << "param_" << param.index;

            // Get motions dimensions
            std::string motions_path = ss_base.str() + "/motions";
            if (H5Lexists(h5file.getId(), motions_path.c_str(), H5P_DEFAULT)) {
                H5::DataSet dataset = h5file.openDataSet(motions_path);
                H5::DataSpace dataspace = dataset.getSpace();
                hsize_t dims[2];
                dataspace.getSimpleExtentDims(dims, nullptr);
                param_samples = static_cast<int>(dims[0]);
            }

            // Get time range
            std::string time_path = ss_base.str() + "/time";
            if (H5Lexists(h5file.getId(), time_path.c_str(), H5P_DEFAULT)) {
                H5::DataSet time_dataset = h5file.openDataSet(time_path);
                std::vector<float> time_data(param_samples);
                time_dataset.read(time_data.data(), H5::PredType::NATIVE_FLOAT);
                param_duration = time_data.back() - time_data.front();
            }

            // Get phase range
            std::string phase_path = ss_base.str() + "/phase";
            if (H5Lexists(h5file.getId(), phase_path.c_str(), H5P_DEFAULT)) {
                H5::DataSet phase_dataset = h5file.openDataSet(phase_path);
                std::vector<float> phase_data(param_samples);
                phase_dataset.read(phase_data.data(), H5::PredType::NATIVE_FLOAT);
                param_phase_min = *std::min_element(phase_data.begin(), phase_data.end());
                param_phase_max = *std::max_element(phase_data.begin(), phase_data.end());
            }

            h5file.close();
        } catch (...) {}

        printw("Data Details:\n");
        printw("-------------\n");
        printw("Frames:    %d\n", param_samples);
        printw("Duration:  %.2f s\n", param_duration);
        printw("Phase:     [%.3f, %.3f]\n", param_phase_min, param_phase_max);
        printw("Frame Rate: 60 Hz\n\n");
    } else {
        CycleInfo& cycle = param.cycles[selected_cycle_idx];

        printw("Cycle Details:\n");
        printw("--------------\n");
        printw("Frames:    %d\n", cycle.frames);
        printw("Duration:  %.2f s\n", cycle.duration);
        printw("Phase:     [%.3f, %.3f]\n", cycle.phase_min, cycle.phase_max);
        printw("Frame Rate: 60 Hz\n\n");
    }

    printw("Output: %s\n", rel_output.string().c_str());
}

bool ExtractorUI::extractDirect(const std::string& filepath, int param_idx, int cycle_idx,
                                const std::string& output_path) {
    // Get relative paths
    fs::path rel_input = fs::relative(filepath);
    fs::path rel_output = fs::relative(output_path);

    std::cout << "Extracting: " << rel_input.string() << std::endl;
    std::cout << "  param_" << param_idx << "/cycle_" << cycle_idx << std::endl;
    std::cout << "  → " << rel_output.string() << std::endl;

    try {
        extractCycle(filepath, param_idx, cycle_idx, output_path);
        std::cout << "Extraction complete!" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

void ExtractorUI::extractCycle(const std::string& input_path, int param_idx, int cycle_idx,
                               const std::string& output_path) {
    try {
        H5::H5File input_file(input_path, H5F_ACC_RDONLY);
        H5::H5File output_file(output_path, H5F_ACC_TRUNC);

        bool is_param_averaged = (cycle_idx == -1);
        std::string base_path;

        if (is_param_averaged) {
            // Extract from param level: /param_X/
            std::stringstream ss;
            ss << "param_" << param_idx;
            base_path = ss.str();
        } else {
            // Extract from cycle level: /param_X/cycle_Y/
            std::stringstream ss;
            ss << "param_" << param_idx << "/cycle_" << cycle_idx;
            base_path = ss.str();
        }

        // Copy motions dataset
        {
            std::string src_path = base_path + "/motions";
            H5::DataSet src_dataset = input_file.openDataSet(src_path);
            H5::DataSpace src_dataspace = src_dataset.getSpace();
            hsize_t dims[2];
            src_dataspace.getSimpleExtentDims(dims, nullptr);

            H5::DataSpace dst_dataspace(2, dims);
            H5::DataSet dst_dataset = output_file.createDataSet("/motions", H5::PredType::NATIVE_FLOAT, dst_dataspace);

            std::vector<float> data(dims[0] * dims[1]);
            src_dataset.read(data.data(), H5::PredType::NATIVE_FLOAT);
            dst_dataset.write(data.data(), H5::PredType::NATIVE_FLOAT);
        }

        // Copy phase dataset
        {
            std::string src_path = base_path + "/phase";
            H5::DataSet src_dataset = input_file.openDataSet(src_path);
            H5::DataSpace src_dataspace = src_dataset.getSpace();
            hsize_t dims[1];
            src_dataspace.getSimpleExtentDims(dims, nullptr);

            H5::DataSpace dst_dataspace(1, dims);
            H5::DataSet dst_dataset = output_file.createDataSet("/phase", H5::PredType::NATIVE_FLOAT, dst_dataspace);

            std::vector<float> data(dims[0]);
            src_dataset.read(data.data(), H5::PredType::NATIVE_FLOAT);
            dst_dataset.write(data.data(), H5::PredType::NATIVE_FLOAT);
        }

        // Copy time dataset
        {
            std::string src_path = base_path + "/time";
            H5::DataSet src_dataset = input_file.openDataSet(src_path);
            H5::DataSpace src_dataspace = src_dataset.getSpace();
            hsize_t dims[1];
            src_dataspace.getSimpleExtentDims(dims, nullptr);

            H5::DataSpace dst_dataspace(1, dims);
            H5::DataSet dst_dataset = output_file.createDataSet("/time", H5::PredType::NATIVE_FLOAT, dst_dataspace);

            std::vector<float> data(dims[0]);
            src_dataset.read(data.data(), H5::PredType::NATIVE_FLOAT);
            dst_dataset.write(data.data(), H5::PredType::NATIVE_FLOAT);
        }

        // Copy param_state dataset
        {
            std::stringstream ss_param;
            ss_param << "param_" << param_idx << "/param_state";
            std::string src_path = ss_param.str();

            if (H5Lexists(input_file.getId(), src_path.c_str(), H5P_DEFAULT)) {
                H5::DataSet src_dataset = input_file.openDataSet(src_path);
                H5::DataSpace src_dataspace = src_dataset.getSpace();
                hsize_t dims[1];
                src_dataspace.getSimpleExtentDims(dims, nullptr);

                H5::DataSpace dst_dataspace(1, dims);
                H5::DataSet dst_dataset = output_file.createDataSet("/param_state", H5::PredType::NATIVE_FLOAT, dst_dataspace);

                std::vector<float> data(dims[0]);
                src_dataset.read(data.data(), H5::PredType::NATIVE_FLOAT);
                dst_dataset.write(data.data(), H5::PredType::NATIVE_FLOAT);
            }
        }

        // Copy param_motions dataset (averaged motion data from param level)
        // Only when extracting a cycle (not when extracting param_averaged itself)
        if (!is_param_averaged) {
            std::stringstream ss_param;
            ss_param << "param_" << param_idx << "/motions";
            std::string src_path = ss_param.str();

            if (H5Lexists(input_file.getId(), src_path.c_str(), H5P_DEFAULT)) {
                H5::DataSet src_dataset = input_file.openDataSet(src_path);
                H5::DataSpace src_dataspace = src_dataset.getSpace();
                hsize_t dims[2];
                src_dataspace.getSimpleExtentDims(dims, nullptr);

                H5::DataSpace dst_dataspace(2, dims);
                H5::DataSet dst_dataset = output_file.createDataSet("/param_motions", H5::PredType::NATIVE_FLOAT, dst_dataspace);

                std::vector<float> data(dims[0] * dims[1]);
                src_dataset.read(data.data(), H5::PredType::NATIVE_FLOAT);
                dst_dataset.write(data.data(), H5::PredType::NATIVE_FLOAT);
            }
        }

        // Create metadata
        {
            std::stringstream metadata;
            metadata << "{\"param_idx\": " << param_idx << ", \"cycle_idx\": " << cycle_idx
                    << ", \"source_file\": \"" << input_path << "\"}";
            std::string metadata_str = metadata.str();

            H5::StrType str_type(H5::PredType::C_S1, metadata_str.size() + 1);
            H5::DataSpace scalar_space(H5S_SCALAR);
            H5::DataSet metadata_dataset = output_file.createDataSet("/metadata", str_type, scalar_space);
            metadata_dataset.write(metadata_str.c_str(), str_type);
        }

        input_file.close();
        output_file.close();

    } catch (const H5::Exception& e) {
        mvprintw(LINES - 4, 0, "Error: %s", e.getDetailMsg().c_str());
        refresh();
        napms(3000);
    }
}

void ExtractorUI::run() {
    initscr();
    noecho();
    cbreak();
    keypad(stdscr, TRUE);

    scanFiles();

    if (files.empty()) {
        printw("No HDF5 files found in sampled/ directory.\n");
        printw("Press any key to exit...");
        refresh();
        getch();
        endwin();
        return;
    }

    while (true) {
        switch (current_stage) {
            case FILE_SELECT:
                stage1_fileSelect();
                break;
            case PARAM_SELECT:
                stage2_paramSelect();
                break;
            case CYCLE_SELECT:
                stage3_cycleSelect();
                break;
            case CONFIRM:
                stageConfirm();
                break;
        }
    }

    endwin();
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [OPTIONS]\n\n";
    std::cout << "Interactive mode (no arguments):\n";
    std::cout << "  " << prog_name << "\n\n";
    std::cout << "Non-interactive mode:\n";
    std::cout << "  " << prog_name << " -f FILE -p PARAM -c CYCLE [-o OUTPUT]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -f, --file FILE       Input HDF5 file path\n";
    std::cout << "  -p, --param PARAM     Parameter index (e.g., 0 for param_0)\n";
    std::cout << "  -c, --cycle CYCLE     Cycle index (e.g., 5 for cycle_5)\n";
    std::cout << "  -o, --output OUTPUT   Output file path (optional, auto-generated if not specified)\n";
    std::cout << "  -h, --help            Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  # Interactive mode\n";
    std::cout << "  " << prog_name << "\n\n";
    std::cout << "  # Extract param_7/cycle_5 from rollout_data.h5\n";
    std::cout << "  " << prog_name << " -f sampled/.../rollout_data.h5 -p 7 -c 5\n\n";
    std::cout << "  # With custom output path\n";
    std::cout << "  " << prog_name << " -f input.h5 -p 0 -c 2 -o output.h5\n";
}

int main(int argc, char** argv) {
    // Command-line argument parsing
    std::string input_file;
    std::string output_file;
    int param_idx = -1;
    int cycle_idx = -1;

    static struct option long_options[] = {
        {"file",   required_argument, 0, 'f'},
        {"param",  required_argument, 0, 'p'},
        {"cycle",  required_argument, 0, 'c'},
        {"output", required_argument, 0, 'o'},
        {"help",   no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "f:p:c:o:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'f':
                input_file = optarg;
                break;
            case 'p':
                param_idx = std::atoi(optarg);
                break;
            case 'c':
                cycle_idx = std::atoi(optarg);
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    ExtractorUI ui;

    // Check if non-interactive mode (all required args provided)
    if (!input_file.empty() && param_idx >= 0 && cycle_idx >= 0) {
        // Non-interactive mode
        if (!fs::exists(input_file)) {
            std::cerr << "Error: Input file not found: " << input_file << std::endl;
            return 1;
        }

        // Generate output filename if not provided
        if (output_file.empty()) {
            // Ensure data/motion directory exists
            fs::create_directories("data/motion");

            fs::path input_path(input_file);
            std::string base_name = input_path.stem().string();
            std::stringstream ss;
            ss << "data/motion/" << base_name
               << "_param" << param_idx << "_cycle" << cycle_idx << ".h5";
            output_file = ss.str();
        }

        bool success = ui.extractDirect(input_file, param_idx, cycle_idx, output_file);
        return success ? 0 : 1;
    }

    // Interactive mode (no arguments or incomplete arguments)
    if (!input_file.empty() || param_idx >= 0 || cycle_idx >= 0) {
        std::cerr << "Error: For non-interactive mode, you must provide -f, -p, and -c\n\n";
        print_usage(argv[0]);
        return 1;
    }

    ui.run();
    return 0;
}
