#include <iostream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <string>
#include <ezc3d/ezc3d_all.h>
#include "rm/rm.hpp"
#include <rm/global.hpp>

namespace fs = std::filesystem;

void processDirectory(rm::ResourceManager& mgr, const std::string& pid, const std::string& prePost) {
    std::string uri = "@pid:" + pid + "/gait/" + prePost;

    // List files and filter for Trimmed_*.c3d
    std::vector<std::string> trimmedFiles;
    try {
        auto files = mgr.list(uri);
        for (const auto& f : files) {
            if (f.find("Trimmed_") == 0 && f.find(".c3d") != std::string::npos
                && f != "Trimmed_unified.c3d") {
                trimmedFiles.push_back(f);
            }
        }
    } catch (const rm::RMError&) {
        return;  // Directory doesn't exist
    }

    if (trimmedFiles.empty()) {
        std::cout << "[" << pid << "/" << prePost << "] No Trimmed_*.c3d files found\n";
        return;
    }

    std::sort(trimmedFiles.begin(), trimmedFiles.end());
    std::cout << "[" << pid << "/" << prePost << "] Merging " << trimmedFiles.size() << " files\n";

    // Load first file to get labels and frame rate
    auto firstHandle = mgr.fetch(uri + "/" + trimmedFiles[0]);
    ezc3d::c3d firstC3d(firstHandle.local_path().string());

    double frameRate = firstC3d.header().frameRate();
    std::vector<std::string> labels;
    try {
        labels = firstC3d.parameters().group("POINT").parameter("LABELS").valuesAsString();
    } catch (...) {}

    // Create output c3d with point labels and frame rate
    ezc3d::c3d output;

    // Set frame rate parameter before adding points
    ezc3d::ParametersNS::GroupNS::Parameter rateParam("RATE");
    rateParam.set(frameRate);
    output.parameter("POINT", rateParam);

    if (!labels.empty()) {
        output.point(labels);
    }

    // Merge all frames from all files
    size_t totalFrames = 0;
    for (const auto& filename : trimmedFiles) {
        auto handle = mgr.fetch(uri + "/" + filename);
        ezc3d::c3d inputC3d(handle.local_path().string());

        size_t numFrames = inputC3d.data().nbFrames();
        for (size_t frameIdx = 0; frameIdx < numFrames; ++frameIdx) {
            const auto& srcFrame = inputC3d.data().frame(frameIdx);
            const auto& srcPoints = srcFrame.points();

            ezc3d::DataNS::Frame outFrame;
            ezc3d::DataNS::Points3dNS::Points outPts(srcPoints.nbPoints());

            for (size_t i = 0; i < srcPoints.nbPoints(); ++i) {
                const auto& srcPt = srcPoints.point(i);
                ezc3d::DataNS::Points3dNS::Point pt;
                pt.set(srcPt.x(), srcPt.y(), srcPt.z());
                outPts.point(pt, i);
            }
            outFrame.add(outPts);
            output.frame(outFrame);
        }
        totalFrames += numFrames;
        std::cout << "  - " << filename << " (" << numFrames << " frames)\n";
    }

    // Write to gait/pre or gait/post directory directly
    fs::path outputPath;

    // Try to fetch existing Trimmed_unified.c3d to get its location
    if (mgr.exists(uri + "/Trimmed_unified.c3d")) {
        auto outputHandle = mgr.fetch(uri + "/Trimmed_unified.c3d");
        outputPath = outputHandle.local_path();
    } else {
        // File doesn't exist yet - determine output path from first source file
        fs::path c3dDir = firstHandle.local_path().parent_path();
        // If source is in Generated_C3D_files, write to parent (gait/pre/)
        if (c3dDir.filename() == "Generated_C3D_files") {
            outputPath = c3dDir.parent_path() / "Trimmed_unified.c3d";
        } else {
            outputPath = c3dDir / "Trimmed_unified.c3d";
        }
    }

    output.write(outputPath.string());
    std::cout << "[" << pid << "/" << prePost << "] Written: " << outputPath
              << " (" << totalFrames << " total frames)\n";
}

int main(int argc, char* argv[]) {
    auto& mgr = rm::getManager();

    if (argc > 1 && std::string(argv[1]) == "--all") {
        auto pids = mgr.list("@pid:");
        std::sort(pids.begin(), pids.end());
        std::cout << "Processing " << pids.size() << " PIDs...\n";
        for (const auto& pid : pids) {
            processDirectory(mgr, pid, "pre");
            processDirectory(mgr, pid, "post");
        }
    } else if (argc > 1) {
        std::string pid = argv[1];
        processDirectory(mgr, pid, "pre");
        processDirectory(mgr, pid, "post");
    } else {
        std::cerr << "Usage: c3d_merge <pid> | --all\n";
        return 1;
    }
    return 0;
}
