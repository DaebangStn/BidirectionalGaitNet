#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <cmath>
#include <set>
#include <filesystem>
#include <locale.h>
#include <ncurses.h>
#include <ezc3d/ezc3d_all.h>
#include "rm/rm.hpp"
#include <rm/global.hpp>

namespace fs = std::filesystem;

// UI State
enum Stage { PID_SELECT, PREPOST_SELECT, FILE_SELECT, INSPECT_VIEW, DIR_BROWSE };

struct C3DInfo {
    std::string uri;
    std::string filename;
    size_t numFrames;
    double frameRate;
    double duration;
    std::vector<std::string> labels;
    std::vector<size_t> nanCounts;  // NaN frame count per marker
    bool loaded;

    C3DInfo() : numFrames(0), frameRate(0), duration(0), loaded(false) {}
};

class C3DInspectorUI {
public:
    C3DInspectorUI(rm::ResourceManager& mgr) : mMgr(mgr), mStage(PID_SELECT),
        mSelectedPID(0), mSelectedPrePost(0), mSelectedFile(0), mScrollOffset(0),
        mEditMode(false), mCursorPos(0), mFirstMarker(-1), mSecondMarker(-1),
        mShowSaveDialog(false), mShowExitConfirm(false), mModified(false),
        mSelectedDirEntry(0), mDirScrollOffset(0),
        mConcatMode(false), mConcatFilter("trimmed_"), mShowConcatDialog(false),
        mBatchProcessing(false),
        mNanInspectMode(false), mNanMarkerCursor(0), mNanMarkerScrollOffset(0),
        mSelectedNanMarker(-1), mNanFrameCursor(0), mNanFrameScrollOffset(0),
        mNanContextFrames(5), mNanShowFrameDetail(false) {}

    void run();

private:
    rm::ResourceManager& mMgr;
    Stage mStage;

    std::vector<std::string> mPIDs;
    std::vector<std::string> mPIDNames;
    std::vector<std::string> mPIDGMFCS;
    std::vector<std::string> mPIDDisplayStrings;
    std::vector<std::string> mPrePost;
    std::vector<std::string> mFiles;
    C3DInfo mC3DInfo;

    int mSelectedPID;
    int mSelectedPrePost;
    int mSelectedFile;
    int mScrollOffset;

    // Edit mode state
    bool mEditMode;
    int mCursorPos;           // current cursor position in marker list
    int mFirstMarker;         // first selected marker (-1 = none)
    int mSecondMarker;        // second selected marker (-1 = none)
    bool mShowSaveDialog;
    bool mShowExitConfirm;    // exit confirmation dialog
    bool mModified;           // true if data was swapped
    std::string mSaveFilename;
    std::string mLocalPath;   // local file path for saving
    std::unique_ptr<ezc3d::c3d> mC3D;  // full c3d object for editing

    // Directory browsing state
    std::string mCurrentDir;
    std::vector<std::string> mDirEntries;      // directory names
    std::vector<std::string> mC3DFiles;        // c3d file names
    int mSelectedDirEntry;
    int mDirScrollOffset;

    // Concatenation state
    bool mConcatMode;
    std::string mConcatFilter;        // Filter pattern (default: "trimmed_")
    std::vector<int> mFilteredIndices; // Indices of files matching filter
    bool mShowConcatDialog;
    bool mBatchProcessing;            // For batch mode progress
    std::string mBatchStatus;         // Status message during batch

    // NaN inspection mode state
    bool mNanInspectMode;
    int mNanMarkerCursor;
    int mNanMarkerScrollOffset;
    std::vector<int> mNanMarkerIndices;      // markers with nanCounts > 0
    int mSelectedNanMarker;
    std::vector<size_t> mNanFrameList;       // frames where selected marker has NaN
    int mNanFrameCursor;
    int mNanFrameScrollOffset;
    int mNanContextFrames;                   // default: 5
    bool mNanShowFrameDetail;
    std::set<size_t> mInterpolatedFrames;   // frames that have been interpolated

    void loadPIDs();
    void loadPIDMetadata();
    void loadFiles();
    void loadC3DInfo();
    void loadDirectory();
    void loadC3DFromPath(const std::string& path);

    void drawPIDSelect();
    void drawPrePostSelect();
    void drawFileSelect();
    void drawInspectView();
    void drawSaveDialog();
    void drawExitConfirm();
    void drawDirBrowse();

    void handleInput(int ch);
    void handleSaveDialogInput(int ch);
    void handleExitConfirmInput(int ch);
    int getMaxVisibleItems();

    void swapMarkerData(int idx1, int idx2);
    void saveC3D();
    std::string getDefaultSaveFilename();

    // Concatenation methods
    void applyFileFilter();
    void drawConcatDialog();
    void handleConcatDialogInput(int ch);
    void concatenateFiles(const std::string& uri, const std::vector<std::string>& files, const std::string& outputName);
    void runBatchMerge();
    void drawBatchProgress(int current, int total, const std::string& currentPid, const std::string& visit);

    // NaN inspection methods
    void buildNanMarkerList();
    void findNanFramesForMarker(int markerIndex);
    void drawNanInspectView();
    void handleNanInspectInput(int ch);
    bool interpolateNanFrame(int markerIndex, size_t nanFrame);
    void dropCurrentNanFrame(size_t frameToRemove);
    void dropAllNanFrames();
    void recalculateNanCounts();
};

void C3DInspectorUI::loadPIDs() {
    mPIDs.clear();
    mPIDNames.clear();
    mPIDGMFCS.clear();
    mPIDDisplayStrings.clear();

    try {
        mPIDs = mMgr.list("@pid:");
        std::sort(mPIDs.begin(), mPIDs.end());
        loadPIDMetadata();
    } catch (const rm::RMError&) {}
}

void C3DInspectorUI::loadPIDMetadata() {
    mPIDNames.resize(mPIDs.size());
    mPIDGMFCS.resize(mPIDs.size());
    mPIDDisplayStrings.resize(mPIDs.size());

    for (size_t i = 0; i < mPIDs.size(); ++i) {
        // Load name
        try {
            std::string nameUri = "@pid:" + mPIDs[i] + "/name";
            auto handle = mMgr.fetch(nameUri);
            mPIDNames[i] = handle.as_string();
        } catch (const rm::RMError&) {
            mPIDNames[i] = "";
        }

        // Load GMFCS
        try {
            std::string gmfcsUri = "@pid:" + mPIDs[i] + "/gmfcs";
            auto handle = mMgr.fetch(gmfcsUri);
            mPIDGMFCS[i] = handle.as_string();
        } catch (const rm::RMError&) {
            mPIDGMFCS[i] = "";
        }

        // Build display string: "12345678 (Name, II)" or "12345678 (Name)" or just "12345678"
        const std::string& pid = mPIDs[i];
        const std::string& name = mPIDNames[i];
        const std::string& gmfcs = mPIDGMFCS[i];

        if (name.empty() && gmfcs.empty()) {
            mPIDDisplayStrings[i] = pid;
        } else if (name.empty()) {
            mPIDDisplayStrings[i] = pid + " (" + gmfcs + ")";
        } else if (gmfcs.empty()) {
            mPIDDisplayStrings[i] = pid + " (" + name + ")";
        } else {
            mPIDDisplayStrings[i] = pid + " (" + name + ", " + gmfcs + ")";
        }
    }
}

void C3DInspectorUI::loadFiles() {
    mFiles.clear();
    if (mPIDs.empty()) return;

    std::string visit = (mSelectedPrePost == 0) ? "pre" : "op1";
    std::string uri = "@pid:" + mPIDs[mSelectedPID] + "/" + visit + "/gait";

    try {
        auto files = mMgr.list(uri);
        for (const auto& f : files) {
            if (f.find(".c3d") != std::string::npos || f.find(".C3D") != std::string::npos) {
                mFiles.push_back(f);
            }
        }
        std::sort(mFiles.begin(), mFiles.end());
    } catch (const rm::RMError&) {}
}

void C3DInspectorUI::loadC3DInfo() {
    mC3DInfo = C3DInfo();
    mC3D.reset();
    mEditMode = false;
    mCursorPos = 0;
    mFirstMarker = -1;
    mSecondMarker = -1;
    mModified = false;
    mShowSaveDialog = false;
    mShowExitConfirm = false;

    // Reset NaN inspection state
    mNanInspectMode = false;
    mNanMarkerCursor = 0;
    mNanMarkerScrollOffset = 0;
    mNanMarkerIndices.clear();
    mSelectedNanMarker = -1;
    mNanFrameList.clear();
    mNanFrameCursor = 0;
    mNanFrameScrollOffset = 0;
    mNanShowFrameDetail = false;
    mInterpolatedFrames.clear();

    if (mFiles.empty()) return;

    std::string visit = (mSelectedPrePost == 0) ? "pre" : "op1";
    mC3DInfo.uri = "@pid:" + mPIDs[mSelectedPID] + "/" + visit + "/gait/" + mFiles[mSelectedFile];
    mC3DInfo.filename = mFiles[mSelectedFile];

    try {
        auto handle = mMgr.fetch(mC3DInfo.uri);
        mLocalPath = handle.local_path().string();
        mC3D = std::make_unique<ezc3d::c3d>(mLocalPath);

        mC3DInfo.numFrames = mC3D->data().nbFrames();
        mC3DInfo.frameRate = mC3D->header().frameRate();
        mC3DInfo.duration = mC3DInfo.numFrames / mC3DInfo.frameRate;

        try {
            mC3DInfo.labels = mC3D->parameters().group("POINT").parameter("LABELS").valuesAsString();
        } catch (...) {}

        // Count NaN frames per marker
        size_t numMarkers = mC3DInfo.labels.size();
        mC3DInfo.nanCounts.resize(numMarkers, 0);
        for (size_t f = 0; f < mC3DInfo.numFrames; ++f) {
            const auto& points = mC3D->data().frame(f).points();
            for (size_t m = 0; m < numMarkers; ++m) {
                const auto& p = points.point(m);
                if (std::isnan(p.x()) || std::isnan(p.y()) || std::isnan(p.z())) {
                    mC3DInfo.nanCounts[m]++;
                }
            }
        }

        mC3DInfo.loaded = true;
    } catch (...) {
        mC3DInfo.loaded = false;
    }
}

int C3DInspectorUI::getMaxVisibleItems() {
    int maxY, maxX;
    getmaxyx(stdscr, maxY, maxX);
    (void)maxX;
    return maxY - 8;  // Leave room for header and footer
}

void C3DInspectorUI::drawPIDSelect() {
    int maxY, maxX;
    getmaxyx(stdscr, maxY, maxX);

    attron(A_BOLD);
    mvprintw(0, 0, "C3D Inspector - Select PID");
    attroff(A_BOLD);
    mvhline(1, 0, '-', maxX);

    int maxVisible = getMaxVisibleItems();
    int startIdx = mScrollOffset;
    int endIdx = std::min(startIdx + maxVisible, (int)mPIDs.size());

    for (int i = startIdx; i < endIdx; ++i) {
        int row = 3 + (i - startIdx);
        const std::string& displayStr = mPIDDisplayStrings[i];
        if (i == mSelectedPID) {
            attron(A_REVERSE);
            mvprintw(row, 2, "%-60s", displayStr.c_str());
            attroff(A_REVERSE);
        } else {
            mvprintw(row, 2, "%-60s", displayStr.c_str());
        }
    }

    mvhline(maxY - 2, 0, '-', maxX);
    mvprintw(maxY - 1, 0, "[UP/DOWN] Navigate  [ENTER] Select  [d] Browse  [m] Merge all  [q] Quit  (%d/%d)",
             mSelectedPID + 1, (int)mPIDs.size());
}

void C3DInspectorUI::drawPrePostSelect() {
    int maxY, maxX;
    getmaxyx(stdscr, maxY, maxX);

    attron(A_BOLD);
    mvprintw(0, 0, "C3D Inspector - PID: %s - Select Pre/Post", mPIDs[mSelectedPID].c_str());
    attroff(A_BOLD);
    mvhline(1, 0, '-', maxX);

    const char* options[] = {"pre", "post"};
    for (int i = 0; i < 2; ++i) {
        int row = 3 + i;
        if (i == mSelectedPrePost) {
            attron(A_REVERSE);
            mvprintw(row, 2, "%-40s", options[i]);
            attroff(A_REVERSE);
        } else {
            mvprintw(row, 2, "%-40s", options[i]);
        }
    }

    mvhline(maxY - 2, 0, '-', maxX);
    mvprintw(maxY - 1, 0, "[UP/DOWN] Navigate  [ENTER] Select  [BACKSPACE] Back  [q] Quit");
}

void C3DInspectorUI::drawFileSelect() {
    int maxY, maxX;
    getmaxyx(stdscr, maxY, maxX);

    std::string prePost = (mSelectedPrePost == 0) ? "pre" : "post";

    attron(A_BOLD);
    mvprintw(0, 0, "C3D Inspector - PID: %s / %s - Select File",
             mPIDs[mSelectedPID].c_str(), prePost.c_str());
    attroff(A_BOLD);
    mvhline(1, 0, '-', maxX);

    if (mFiles.empty()) {
        mvprintw(3, 2, "(no C3D files found)");
    } else {
        int maxVisible = getMaxVisibleItems();
        int startIdx = mScrollOffset;
        int endIdx = std::min(startIdx + maxVisible, (int)mFiles.size());

        for (int i = startIdx; i < endIdx; ++i) {
            int row = 3 + (i - startIdx);
            if (i == mSelectedFile) {
                attron(A_REVERSE);
                mvprintw(row, 2, "%-60s", mFiles[i].c_str());
                attroff(A_REVERSE);
            } else {
                mvprintw(row, 2, "%-60s", mFiles[i].c_str());
            }
        }
    }

    mvhline(maxY - 2, 0, '-', maxX);
    mvprintw(maxY - 1, 0, "[UP/DOWN] Navigate  [ENTER] Select  [c] Concatenate  [BACKSPACE] Back  [q] Quit  (%d/%d)",
             mFiles.empty() ? 0 : mSelectedFile + 1, (int)mFiles.size());
}

void C3DInspectorUI::drawInspectView() {
    int maxY, maxX;
    getmaxyx(stdscr, maxY, maxX);

    // Header
    attron(A_BOLD);
    if (mEditMode) {
        mvprintw(0, 0, "C3D Inspector - %s [EDIT MODE]%s",
                 mC3DInfo.filename.c_str(), mModified ? " *" : "");
    } else {
        mvprintw(0, 0, "C3D Inspector - %s%s",
                 mC3DInfo.filename.c_str(), mModified ? " *" : "");
    }
    attroff(A_BOLD);
    mvhline(1, 0, '-', maxX);

    if (!mC3DInfo.loaded) {
        mvprintw(3, 2, "Error loading C3D file");
    } else {
        mvprintw(3, 2, "URI: %s", mC3DInfo.uri.c_str());
        mvprintw(5, 2, "Total Frames: %zu", mC3DInfo.numFrames);
        mvprintw(6, 2, "Frame Rate:   %.1f Hz", mC3DInfo.frameRate);
        mvprintw(7, 2, "Duration:     %.2f s", mC3DInfo.duration);

        mvprintw(9, 2, "Labels (%zu markers):", mC3DInfo.labels.size());
        mvhline(10, 2, '-', 50);

        int maxVisible = maxY - 14;
        int startIdx = mScrollOffset;
        int endIdx = std::min(startIdx + maxVisible, (int)mC3DInfo.labels.size());

        for (int i = startIdx; i < endIdx; ++i) {
            int row = 11 + (i - startIdx);

            // Determine highlighting
            bool isFirstMarker = (i == mFirstMarker);
            bool isSecondMarker = (i == mSecondMarker);
            bool isCursor = mEditMode && (i == mCursorPos);
            bool hasNaN = (i < (int)mC3DInfo.nanCounts.size()) && (mC3DInfo.nanCounts[i] > 0);

            if (isFirstMarker) {
                attron(A_BOLD | A_UNDERLINE);
                mvprintw(row, 2, "%4d: %s", i, mC3DInfo.labels[i].c_str());
                attroff(A_BOLD | A_UNDERLINE);
                if (hasNaN) {
                    attron(COLOR_PAIR(2));  // Red for NaN count
                    printw(" (%zu NaN)", mC3DInfo.nanCounts[i]);
                    attroff(COLOR_PAIR(2));
                }
                printw(" [1st]");
            } else if (isSecondMarker) {
                attron(A_BOLD | A_UNDERLINE);
                mvprintw(row, 2, "%4d: %s", i, mC3DInfo.labels[i].c_str());
                attroff(A_BOLD | A_UNDERLINE);
                if (hasNaN) {
                    attron(COLOR_PAIR(2));
                    printw(" (%zu NaN)", mC3DInfo.nanCounts[i]);
                    attroff(COLOR_PAIR(2));
                }
                printw(" [2nd]");
            } else if (isCursor) {
                attron(A_REVERSE);
                if (hasNaN) {
                    mvprintw(row, 2, "%4d: %-20s", i, mC3DInfo.labels[i].c_str());
                    attroff(A_REVERSE);
                    attron(COLOR_PAIR(2));
                    printw(" (%zu NaN)", mC3DInfo.nanCounts[i]);
                    attroff(COLOR_PAIR(2));
                } else {
                    mvprintw(row, 2, "%4d: %-40s", i, mC3DInfo.labels[i].c_str());
                    attroff(A_REVERSE);
                }
            } else {
                if (hasNaN) {
                    attron(COLOR_PAIR(2));  // Red for markers with NaN
                    mvprintw(row, 2, "%4d: %s (%zu NaN)", i, mC3DInfo.labels[i].c_str(), mC3DInfo.nanCounts[i]);
                    attroff(COLOR_PAIR(2));
                } else {
                    attron(COLOR_PAIR(1));  // Green for markers without NaN
                    mvprintw(row, 2, "%4d: %s", i, mC3DInfo.labels[i].c_str());
                    attroff(COLOR_PAIR(1));
                }
            }
        }

        // Show swap prompt when both markers selected
        if (mEditMode && mFirstMarker >= 0 && mSecondMarker >= 0) {
            mvprintw(maxY - 4, 2, "Swap '%s' <-> '%s'?  [y] Yes  [n] No",
                     mC3DInfo.labels[mFirstMarker].c_str(),
                     mC3DInfo.labels[mSecondMarker].c_str());
        }
    }

    // Footer
    mvhline(maxY - 2, 0, '-', maxX);
    if (mEditMode) {
        if (mFirstMarker < 0) {
            mvprintw(maxY - 1, 0, "[UP/DOWN] Navigate  [ENTER] Select 1st marker  [ESC] Exit edit  (%d/%d)",
                     mCursorPos + 1, (int)mC3DInfo.labels.size());
        } else if (mSecondMarker < 0) {
            mvprintw(maxY - 1, 0, "[UP/DOWN] Navigate  [ENTER] Select 2nd marker  [ESC] Exit edit  (%d/%d)",
                     mCursorPos + 1, (int)mC3DInfo.labels.size());
        } else {
            mvprintw(maxY - 1, 0, "[y] Swap  [n] Cancel  [ESC] Exit edit");
        }
    } else {
        if (mModified) {
            mvprintw(maxY - 1, 0, "[UP/DOWN] Scroll  [e] Edit  [n] NaN inspect  [s] Save  [BACKSPACE] Back  [q] Quit  (%d/%d)",
                     mC3DInfo.labels.empty() ? 0 : mScrollOffset + 1, (int)mC3DInfo.labels.size());
        } else {
            mvprintw(maxY - 1, 0, "[UP/DOWN] Scroll  [e] Edit  [n] NaN inspect  [BACKSPACE] Back  [q] Quit  (%d/%d)",
                     mC3DInfo.labels.empty() ? 0 : mScrollOffset + 1, (int)mC3DInfo.labels.size());
        }
    }
}

std::string C3DInspectorUI::getDefaultSaveFilename() {
    std::string filename = mC3DInfo.filename;
    // Remove .c3d or .C3D extension
    size_t pos = filename.rfind(".c3d");
    if (pos == std::string::npos) pos = filename.rfind(".C3D");
    if (pos != std::string::npos) {
        filename = filename.substr(0, pos);
    }
    return filename + ".c3d";
}

void C3DInspectorUI::swapMarkerData(int idx1, int idx2) {
    if (!mC3D || idx1 < 0 || idx2 < 0) return;

    size_t numFrames = mC3D->data().nbFrames();
    for (size_t f = 0; f < numFrames; ++f) {
        // Copy the original frame
        ezc3d::DataNS::Frame newFrame = mC3D->data().frame(f);

        // Get references to modifiable points
        auto& points = newFrame.points();
        auto& p1 = points.point(idx1);
        auto& p2 = points.point(idx2);

        // Store original values
        double x1 = p1.x(), y1 = p1.y(), z1 = p1.z(), r1 = p1.residual();
        double x2 = p2.x(), y2 = p2.y(), z2 = p2.z(), r2 = p2.residual();

        // Swap
        p1.x(x2); p1.y(y2); p1.z(z2); p1.residual(r2);
        p2.x(x1); p2.y(y1); p2.z(z1); p2.residual(r1);

        // Replace the frame in the c3d
        mC3D->frame(newFrame, f);
    }

    mModified = true;
}

void C3DInspectorUI::saveC3D() {
    if (!mC3D || mSaveFilename.empty()) return;

    // Get directory from local path
    std::string dir = mLocalPath;
    size_t lastSlash = dir.rfind('/');
    if (lastSlash != std::string::npos) {
        dir = dir.substr(0, lastSlash + 1);
    } else {
        dir = "./";
    }

    std::string savePath = dir + mSaveFilename;
    mC3D->write(savePath);
}

void C3DInspectorUI::applyFileFilter() {
    mFilteredIndices.clear();

    // Case-insensitive filter matching
    std::string filterLower = mConcatFilter;
    std::transform(filterLower.begin(), filterLower.end(), filterLower.begin(), ::tolower);

    for (size_t i = 0; i < mFiles.size(); ++i) {
        std::string fileLower = mFiles[i];
        std::transform(fileLower.begin(), fileLower.end(), fileLower.begin(), ::tolower);

        // Match files starting with filter and exclude *_unified.c3d
        if (fileLower.find(filterLower) == 0 &&
            fileLower.find("_unified.c3d") == std::string::npos) {
            mFilteredIndices.push_back(i);
        }
    }
}

void C3DInspectorUI::drawConcatDialog() {
    int maxY, maxX;
    getmaxyx(stdscr, maxY, maxX);

    int dialogWidth = 60;
    int dialogHeight = 9;
    int startX = (maxX - dialogWidth) / 2;
    int startY = (maxY - dialogHeight) / 2;

    // Clear dialog area
    for (int y = startY; y < startY + dialogHeight; ++y) {
        mvhline(y, startX, ' ', dialogWidth);
    }

    // Border
    mvhline(startY, startX, '-', dialogWidth);
    mvhline(startY + dialogHeight - 1, startX, '-', dialogWidth);
    for (int y = startY; y < startY + dialogHeight; ++y) {
        mvaddch(y, startX, '|');
        mvaddch(y, startX + dialogWidth - 1, '|');
    }

    // Content
    attron(A_BOLD);
    mvprintw(startY + 1, startX + 2, "Concatenate C3D Files");
    attroff(A_BOLD);

    mvprintw(startY + 3, startX + 2, "Filter pattern:");
    attron(A_REVERSE);
    mvprintw(startY + 3, startX + 18, "%-38s", mConcatFilter.c_str());
    attroff(A_REVERSE);

    mvprintw(startY + 5, startX + 2, "Matching files: %zu", mFilteredIndices.size());

    // Output filename
    std::string outputName = mConcatFilter + "_unified.c3d";
    mvprintw(startY + 6, startX + 2, "Output: %s", outputName.c_str());

    mvprintw(startY + 7, startX + 2, "[ENTER] Merge  [ESC] Cancel");
}

void C3DInspectorUI::handleConcatDialogInput(int ch) {
    if (ch == 27) {  // ESC
        mShowConcatDialog = false;
    } else if (ch == '\n' || ch == KEY_ENTER) {
        if (!mFilteredIndices.empty()) {
            // Build file list and concatenate
            std::string visit = (mSelectedPrePost == 0) ? "pre" : "op1";
            std::string uri = "@pid:" + mPIDs[mSelectedPID] + "/" + visit + "/gait";

            std::vector<std::string> filesToMerge;
            for (int idx : mFilteredIndices) {
                filesToMerge.push_back(mFiles[idx]);
            }
            std::sort(filesToMerge.begin(), filesToMerge.end());

            std::string outputName = mConcatFilter + "_unified.c3d";
            concatenateFiles(uri, filesToMerge, outputName);

            // Reload file list to show new file
            loadFiles();
        }
        mShowConcatDialog = false;
    } else if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
        if (!mConcatFilter.empty()) {
            mConcatFilter.pop_back();
            applyFileFilter();
        }
    } else if (ch >= 32 && ch < 127) {  // Printable ASCII
        if (mConcatFilter.length() < 30) {
            mConcatFilter += static_cast<char>(ch);
            applyFileFilter();
        }
    }
}

void C3DInspectorUI::concatenateFiles(const std::string& uri,
                                       const std::vector<std::string>& files,
                                       const std::string& outputName) {
    if (files.empty()) return;

    // Load first file to get labels and frame rate
    auto firstHandle = mMgr.fetch(uri + "/" + files[0]);
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
    for (const auto& filename : files) {
        auto handle = mMgr.fetch(uri + "/" + filename);
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
    }

    // Write to gait directory
    fs::path outputPath;

    // Check if output file already exists
    if (mMgr.exists(uri + "/" + outputName)) {
        auto outputHandle = mMgr.fetch(uri + "/" + outputName);
        outputPath = outputHandle.local_path();
    } else {
        // Determine output path from first source file
        fs::path c3dDir = firstHandle.local_path().parent_path();
        // If source is in Generated_C3D_files, write to parent ({visit}/gait/)
        if (c3dDir.filename() == "Generated_C3D_files") {
            outputPath = c3dDir.parent_path() / outputName;
        } else {
            outputPath = c3dDir / outputName;
        }
    }

    output.write(outputPath.string());
}

void C3DInspectorUI::drawBatchProgress(int current, int total, const std::string& currentPid, const std::string& visit) {
    int maxY, maxX;
    getmaxyx(stdscr, maxY, maxX);

    int dialogWidth = 60;
    int dialogHeight = 9;
    int startX = (maxX - dialogWidth) / 2;
    int startY = (maxY - dialogHeight) / 2;

    // Clear dialog area
    for (int y = startY; y < startY + dialogHeight; ++y) {
        mvhline(y, startX, ' ', dialogWidth);
    }

    // Border
    mvhline(startY, startX, '-', dialogWidth);
    mvhline(startY + dialogHeight - 1, startX, '-', dialogWidth);
    for (int y = startY; y < startY + dialogHeight; ++y) {
        mvaddch(y, startX, '|');
        mvaddch(y, startX + dialogWidth - 1, '|');
    }

    // Title
    attron(A_BOLD);
    mvprintw(startY + 1, startX + 2, "Batch Merge Progress");
    attroff(A_BOLD);

    // Current item
    mvprintw(startY + 3, startX + 2, "Processing: %s / %s", currentPid.c_str(), visit.c_str());

    // Progress bar
    int barWidth = dialogWidth - 8;
    float progress = (total > 0) ? (float)current / total : 0.0f;
    int filledWidth = (int)(progress * barWidth);

    mvprintw(startY + 5, startX + 2, "[");
    attron(A_REVERSE);
    for (int i = 0; i < filledWidth; ++i) {
        mvaddch(startY + 5, startX + 3 + i, ' ');
    }
    attroff(A_REVERSE);
    for (int i = filledWidth; i < barWidth; ++i) {
        mvaddch(startY + 5, startX + 3 + i, '-');
    }
    mvprintw(startY + 5, startX + 3 + barWidth, "]");

    // Percentage and count
    mvprintw(startY + 7, startX + 2, "%d / %d  (%.0f%%)", current, total, progress * 100);

    refresh();
}

void C3DInspectorUI::runBatchMerge() {
    mBatchProcessing = true;

    std::string filterLower = "trimmed_";

    // Calculate total operations (each PID has 2 visits)
    int total = mPIDs.size() * 2;
    int current = 0;

    for (const auto& pid : mPIDs) {
        for (const auto& visit : {"pre", "op1"}) {
            // Update progress display
            drawBatchProgress(current, total, pid, visit);

            std::string uri = "@pid:" + pid + "/" + visit + "/gait";

            // List and filter files
            std::vector<std::string> trimmedFiles;
            try {
                auto files = mMgr.list(uri);
                for (const auto& f : files) {
                    std::string fLower = f;
                    std::transform(fLower.begin(), fLower.end(), fLower.begin(), ::tolower);

                    if (fLower.find(filterLower) == 0 &&
                        fLower.find("_unified.c3d") == std::string::npos) {
                        trimmedFiles.push_back(f);
                    }
                }
            } catch (const rm::RMError&) {
                current++;
                continue;  // Directory doesn't exist
            }

            if (!trimmedFiles.empty()) {
                std::sort(trimmedFiles.begin(), trimmedFiles.end());
                concatenateFiles(uri, trimmedFiles, "trimmed_unified.c3d");
            }

            current++;
        }
    }

    // Show completion
    drawBatchProgress(total, total, "Complete", "");

    mBatchProcessing = false;
    mBatchStatus = "Batch merge complete";
}


void C3DInspectorUI::buildNanMarkerList() {
    mNanMarkerIndices.clear();
    for (size_t i = 0; i < mC3DInfo.nanCounts.size(); ++i) {
        if (mC3DInfo.nanCounts[i] > 0) {
            mNanMarkerIndices.push_back((int)i);
        }
    }
}

void C3DInspectorUI::findNanFramesForMarker(int markerIndex) {
    mNanFrameList.clear();
    if (!mC3D || markerIndex < 0 || markerIndex >= (int)mC3DInfo.labels.size()) return;

    for (size_t f = 0; f < mC3DInfo.numFrames; ++f) {
        const auto& points = mC3D->data().frame(f).points();
        const auto& p = points.point(markerIndex);
        if (std::isnan(p.x()) || std::isnan(p.y()) || std::isnan(p.z())) {
            mNanFrameList.push_back(f);
        }
    }
}

bool C3DInspectorUI::interpolateNanFrame(int markerIndex, size_t nanFrame) {
    if (!mC3D || markerIndex < 0 || markerIndex >= (int)mC3DInfo.labels.size()) {
        return false;
    }
    if (nanFrame >= mC3DInfo.numFrames) return false;

    // Find nearest valid frame BEFORE nanFrame
    int beforeFrame = -1;
    for (int f = (int)nanFrame - 1; f >= 0; --f) {
        const auto& points = mC3D->data().frame(f).points();
        const auto& p = points.point(markerIndex);
        if (!std::isnan(p.x()) && !std::isnan(p.y()) && !std::isnan(p.z())) {
            beforeFrame = f;
            break;
        }
    }

    // Find nearest valid frame AFTER nanFrame
    int afterFrame = -1;
    for (size_t f = nanFrame + 1; f < mC3DInfo.numFrames; ++f) {
        const auto& points = mC3D->data().frame(f).points();
        const auto& p = points.point(markerIndex);
        if (!std::isnan(p.x()) && !std::isnan(p.y()) && !std::isnan(p.z())) {
            afterFrame = (int)f;
            break;
        }
    }

    // Determine interpolation values
    double newX, newY, newZ;
    if (beforeFrame >= 0 && afterFrame >= 0) {
        // Linear interpolation between before and after
        const auto& pBefore = mC3D->data().frame(beforeFrame).points().point(markerIndex);
        const auto& pAfter = mC3D->data().frame(afterFrame).points().point(markerIndex);
        double t = (double)(nanFrame - beforeFrame) / (double)(afterFrame - beforeFrame);
        newX = pBefore.x() + t * (pAfter.x() - pBefore.x());
        newY = pBefore.y() + t * (pAfter.y() - pBefore.y());
        newZ = pBefore.z() + t * (pAfter.z() - pBefore.z());
    } else if (beforeFrame >= 0) {
        // Extrapolate from before (just copy)
        const auto& pBefore = mC3D->data().frame(beforeFrame).points().point(markerIndex);
        newX = pBefore.x();
        newY = pBefore.y();
        newZ = pBefore.z();
    } else if (afterFrame >= 0) {
        // Extrapolate from after (just copy)
        const auto& pAfter = mC3D->data().frame(afterFrame).points().point(markerIndex);
        newX = pAfter.x();
        newY = pAfter.y();
        newZ = pAfter.z();
    } else {
        // No valid frames found, cannot interpolate
        return false;
    }

    // Update the C3D data
    ezc3d::DataNS::Frame newFrame = mC3D->data().frame(nanFrame);
    auto& points = newFrame.points();
    auto& p = points.point(markerIndex);
    p.x(newX);
    p.y(newY);
    p.z(newZ);
    mC3D->frame(newFrame, nanFrame);

    // Track as interpolated for visual distinction
    mInterpolatedFrames.insert(nanFrame);
    mModified = true;
    return true;
}

void C3DInspectorUI::recalculateNanCounts() {
    size_t numMarkers = mC3DInfo.labels.size();
    mC3DInfo.nanCounts.assign(numMarkers, 0);
    for (size_t f = 0; f < mC3DInfo.numFrames; ++f) {
        const auto& points = mC3D->data().frame(f).points();
        for (size_t m = 0; m < numMarkers; ++m) {
            const auto& p = points.point(m);
            if (std::isnan(p.x()) || std::isnan(p.y()) || std::isnan(p.z())) {
                mC3DInfo.nanCounts[m]++;
            }
        }
    }
}

void C3DInspectorUI::dropCurrentNanFrame(size_t frameToRemove) {
    if (!mC3D || frameToRemove >= mC3DInfo.numFrames) return;

    // Get labels and frame rate from current C3D
    double frameRate = mC3D->header().frameRate();
    std::vector<std::string> labels = mC3DInfo.labels;

    // Create new C3D with only the frames we want to keep
    ezc3d::c3d newC3D;

    // Set frame rate parameter
    ezc3d::ParametersNS::GroupNS::Parameter rateParam("RATE");
    rateParam.set(frameRate);
    newC3D.parameter("POINT", rateParam);

    if (!labels.empty()) {
        newC3D.point(labels);
    }

    // Copy all frames except the one to remove
    for (size_t f = 0; f < mC3DInfo.numFrames; ++f) {
        if (f != frameToRemove) {
            const auto& srcFrame = mC3D->data().frame(f);
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
            newC3D.frame(outFrame);
        }
    }

    // Replace current C3D
    *mC3D = std::move(newC3D);

    // Update state
    mC3DInfo.numFrames = mC3D->data().nbFrames();
    mC3DInfo.duration = mC3DInfo.numFrames / mC3DInfo.frameRate;

    // Adjust interpolated frames tracking (shift indices)
    std::set<size_t> newInterpolated;
    for (size_t f : mInterpolatedFrames) {
        if (f < frameToRemove) {
            newInterpolated.insert(f);
        } else if (f > frameToRemove) {
            newInterpolated.insert(f - 1);
        }
    }
    mInterpolatedFrames = std::move(newInterpolated);

    recalculateNanCounts();
    mModified = true;
}

void C3DInspectorUI::dropAllNanFrames() {
    if (!mC3D) return;

    // Collect all frames that have ANY NaN in ANY marker
    std::set<size_t> framesToDrop;
    for (size_t f = 0; f < mC3DInfo.numFrames; ++f) {
        const auto& points = mC3D->data().frame(f).points();
        for (size_t m = 0; m < mC3DInfo.labels.size(); ++m) {
            const auto& p = points.point(m);
            if (std::isnan(p.x()) || std::isnan(p.y()) || std::isnan(p.z())) {
                framesToDrop.insert(f);
                break;
            }
        }
    }

    if (framesToDrop.empty()) return;

    // Get labels and frame rate from current C3D
    double frameRate = mC3D->header().frameRate();
    std::vector<std::string> labels = mC3DInfo.labels;

    // Create new C3D with only valid frames
    ezc3d::c3d newC3D;

    // Set frame rate parameter
    ezc3d::ParametersNS::GroupNS::Parameter rateParam("RATE");
    rateParam.set(frameRate);
    newC3D.parameter("POINT", rateParam);

    if (!labels.empty()) {
        newC3D.point(labels);
    }

    // Add only non-NaN frames
    for (size_t f = 0; f < mC3DInfo.numFrames; ++f) {
        if (framesToDrop.count(f) == 0) {
            const auto& srcFrame = mC3D->data().frame(f);
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
            newC3D.frame(outFrame);
        }
    }

    // Replace current C3D
    *mC3D = std::move(newC3D);

    // Update state
    mC3DInfo.numFrames = mC3D->data().nbFrames();
    mC3DInfo.duration = mC3DInfo.numFrames / mC3DInfo.frameRate;

    // Clear interpolated frames (all shifted)
    mInterpolatedFrames.clear();

    // Recalculate NaN counts (should be 0 now)
    std::fill(mC3DInfo.nanCounts.begin(), mC3DInfo.nanCounts.end(), 0);

    mModified = true;
}

void C3DInspectorUI::drawNanInspectView() {
    int maxY, maxX;
    getmaxyx(stdscr, maxY, maxX);

    // Header
    attron(A_BOLD);
    if (mNanShowFrameDetail) {
        mvprintw(0, 0, "NaN Inspection - %s [Frame Detail]",
                 mC3DInfo.labels[mSelectedNanMarker].c_str());
    } else {
        mvprintw(0, 0, "NaN Inspection - %s", mC3DInfo.filename.c_str());
    }
    attroff(A_BOLD);
    mvhline(1, 0, '-', maxX);

    if (mNanShowFrameDetail) {
        // Frame detail view
        if (mNanFrameList.empty()) {
            mvprintw(3, 2, "No NaN frames for this marker");
        } else {
            int currentNanIndex = mNanFrameCursor;
            size_t nanFrame = mNanFrameList[currentNanIndex];

            mvprintw(3, 2, ">>> NaN #%d at frame %zu <<<", currentNanIndex + 1, nanFrame);
            mvprintw(4, 2, "Marker: %s (index %d)", 
                     mC3DInfo.labels[mSelectedNanMarker].c_str(), mSelectedNanMarker);
            mvprintw(5, 2, "Context: +/- %d frames", mNanContextFrames);

            // Table header
            mvprintw(7, 2, "%-8s  %12s  %12s  %12s  %s", "Frame", "X", "Y", "Z", "Status");
            mvhline(8, 2, '-', 60);

            // Calculate frame range
            int startFrame = std::max(0, (int)nanFrame - mNanContextFrames);
            int endFrame = std::min((int)mC3DInfo.numFrames - 1, (int)nanFrame + mNanContextFrames);

            int maxVisibleFrames = maxY - 12;
            int displayStart = startFrame;
            int displayEnd = std::min(endFrame, displayStart + maxVisibleFrames - 1);

            for (int f = displayStart; f <= displayEnd; ++f) {
                int row = 9 + (f - displayStart);
                const auto& points = mC3D->data().frame(f).points();
                const auto& p = points.point(mSelectedNanMarker);

                bool isNan = std::isnan(p.x()) || std::isnan(p.y()) || std::isnan(p.z());
                bool isCurrentNan = ((size_t)f == nanFrame);
                bool wasInterpolated = mInterpolatedFrames.count((size_t)f) > 0;

                if (isCurrentNan) {
                    attron(A_BOLD | COLOR_PAIR(2));
                } else if (wasInterpolated) {
                    attron(COLOR_PAIR(3));  // Green for interpolated
                }

                if (isNan) {
                    mvprintw(row, 2, "%-8d  %12s  %12s  %12s  %s",
                             f,
                             std::isnan(p.x()) ? "NaN" : std::to_string(p.x()).substr(0, 10).c_str(),
                             std::isnan(p.y()) ? "NaN" : std::to_string(p.y()).substr(0, 10).c_str(),
                             std::isnan(p.z()) ? "NaN" : std::to_string(p.z()).substr(0, 10).c_str(),
                             isCurrentNan ? "<-- NaN" : "");
                } else if (wasInterpolated) {
                    mvprintw(row, 2, "%-8d [%11.3f] [%11.3f] [%11.3f]  <-- interp",
                             f, p.x(), p.y(), p.z());
                } else {
                    mvprintw(row, 2, "%-8d  %12.3f  %12.3f  %12.3f",
                             f, p.x(), p.y(), p.z());
                }

                if (isCurrentNan) {
                    attroff(A_BOLD | COLOR_PAIR(2));
                } else if (wasInterpolated) {
                    attroff(COLOR_PAIR(3));
                }
            }
        }

        // Footer for frame detail view
        mvhline(maxY - 2, 0, '-', maxX);
        mvprintw(maxY - 1, 0, "[UP/DN] Nav  [i] Interp  [d] Drop frame  [+/-] Ctx (%d)  [ESC] Back  (%d/%zu)",
                 mNanContextFrames, mNanFrameCursor + 1, mNanFrameList.size());
    } else {
        // Marker list view
        if (mNanMarkerIndices.empty()) {
            mvprintw(3, 2, "No markers with NaN values found");
        } else {
            mvprintw(3, 2, "Markers with NaN values (%zu):", mNanMarkerIndices.size());
            mvhline(4, 2, '-', 50);

            int maxVisible = maxY - 8;
            int startIdx = mNanMarkerScrollOffset;
            int endIdx = std::min(startIdx + maxVisible, (int)mNanMarkerIndices.size());

            for (int i = startIdx; i < endIdx; ++i) {
                int row = 5 + (i - startIdx);
                int markerIdx = mNanMarkerIndices[i];
                bool isCursor = (i == mNanMarkerCursor);

                if (isCursor) {
                    attron(A_REVERSE);
                }

                attron(COLOR_PAIR(2));  // Red for NaN markers
                mvprintw(row, 2, "%4d: %-30s (%zu NaN frames)",
                         markerIdx, mC3DInfo.labels[markerIdx].c_str(), mC3DInfo.nanCounts[markerIdx]);
                attroff(COLOR_PAIR(2));

                if (isCursor) {
                    attroff(A_REVERSE);
                }
            }
        }

        // Footer for marker list view
        mvhline(maxY - 2, 0, '-', maxX);
        mvprintw(maxY - 1, 0, "[UP/DN] Nav  [ENTER] View  [d] Drop ALL NaN frames  [ESC] Back  (%d/%zu)",
                 mNanMarkerIndices.empty() ? 0 : mNanMarkerCursor + 1, mNanMarkerIndices.size());
    }
}

void C3DInspectorUI::handleNanInspectInput(int ch) {
    int maxY, maxX;
    getmaxyx(stdscr, maxY, maxX);
    (void)maxX;

    if (mNanShowFrameDetail) {
        // Frame detail view navigation
        if (ch == KEY_UP && mNanFrameCursor > 0) {
            mNanFrameCursor--;
        } else if (ch == KEY_DOWN && mNanFrameCursor < (int)mNanFrameList.size() - 1) {
            mNanFrameCursor++;
        } else if (ch == '+' || ch == '=') {
            if (mNanContextFrames < 20) mNanContextFrames++;
        } else if (ch == '-' || ch == '_') {
            if (mNanContextFrames > 1) mNanContextFrames--;
        } else if (ch == 'i' || ch == 'I') {
            // Interpolate current NaN frame for this marker
            if (!mNanFrameList.empty()) {
                size_t nanFrame = mNanFrameList[mNanFrameCursor];
                if (interpolateNanFrame(mSelectedNanMarker, nanFrame)) {
                    // Update nanCounts and NaN frame list
                    mC3DInfo.nanCounts[mSelectedNanMarker]--;
                    mNanFrameList.erase(mNanFrameList.begin() + mNanFrameCursor);
                    if (mNanFrameCursor >= (int)mNanFrameList.size()) {
                        mNanFrameCursor = std::max(0, (int)mNanFrameList.size() - 1);
                    }
                    // If no more NaN frames for this marker, go back to marker list
                    if (mNanFrameList.empty()) {
                        mNanShowFrameDetail = false;
                        buildNanMarkerList();
                        if (mNanMarkerIndices.empty()) {
                            mNanInspectMode = false;
                        }
                    }
                }
            }
        } else if (ch == 'd' || ch == 'D') {
            // Drop current frame entirely (all markers)
            if (!mNanFrameList.empty()) {
                size_t frameToRemove = mNanFrameList[mNanFrameCursor];
                dropCurrentNanFrame(frameToRemove);
                findNanFramesForMarker(mSelectedNanMarker);  // Rebuild frame list
                if (mNanFrameList.empty()) {
                    mNanShowFrameDetail = false;
                    buildNanMarkerList();
                    if (mNanMarkerIndices.empty()) {
                        mNanInspectMode = false;
                    }
                } else {
                    mNanFrameCursor = std::min(mNanFrameCursor, (int)mNanFrameList.size() - 1);
                }
            }
        } else if (ch == 27) {  // ESC
            mNanShowFrameDetail = false;
        }
    } else {
        // Marker list navigation
        int maxVisibleMarkers = maxY - 8;

        if (ch == KEY_UP && mNanMarkerCursor > 0) {
            mNanMarkerCursor--;
            if (mNanMarkerCursor < mNanMarkerScrollOffset) {
                mNanMarkerScrollOffset = mNanMarkerCursor;
            }
        } else if (ch == KEY_DOWN && mNanMarkerCursor < (int)mNanMarkerIndices.size() - 1) {
            mNanMarkerCursor++;
            if (mNanMarkerCursor >= mNanMarkerScrollOffset + maxVisibleMarkers) {
                mNanMarkerScrollOffset = mNanMarkerCursor - maxVisibleMarkers + 1;
            }
        } else if ((ch == '\n' || ch == KEY_ENTER) && !mNanMarkerIndices.empty()) {
            mSelectedNanMarker = mNanMarkerIndices[mNanMarkerCursor];
            findNanFramesForMarker(mSelectedNanMarker);
            mNanFrameCursor = 0;
            mNanFrameScrollOffset = 0;
            mNanShowFrameDetail = true;
        } else if (ch == 'd' || ch == 'D') {
            // Drop ALL frames containing any NaN
            dropAllNanFrames();
            buildNanMarkerList();  // Rebuild (should be empty now)
            if (mNanMarkerIndices.empty()) {
                mNanInspectMode = false;  // Exit NaN mode, no more NaNs
            }
        } else if (ch == 27) {  // ESC
            mNanInspectMode = false;
        }
    }
}

void C3DInspectorUI::drawSaveDialog() {
    int maxY, maxX;
    getmaxyx(stdscr, maxY, maxX);

    int dialogWidth = 50;
    int dialogHeight = 7;
    int startX = (maxX - dialogWidth) / 2;
    int startY = (maxY - dialogHeight) / 2;

    // Draw dialog box
    for (int y = startY; y < startY + dialogHeight; ++y) {
        mvhline(y, startX, ' ', dialogWidth);
    }

    // Border
    mvhline(startY, startX, '-', dialogWidth);
    mvhline(startY + dialogHeight - 1, startX, '-', dialogWidth);
    for (int y = startY; y < startY + dialogHeight; ++y) {
        mvaddch(y, startX, '|');
        mvaddch(y, startX + dialogWidth - 1, '|');
    }

    // Content
    attron(A_BOLD);
    mvprintw(startY + 1, startX + 2, "Save as:");
    attroff(A_BOLD);

    // Input field with cursor
    attron(A_REVERSE);
    mvprintw(startY + 3, startX + 2, "%-44s", mSaveFilename.c_str());
    attroff(A_REVERSE);

    mvprintw(startY + 5, startX + 2, "[ENTER] Save  [ESC] Cancel");
}

void C3DInspectorUI::handleSaveDialogInput(int ch) {
    if (ch == 27) {  // ESC
        mShowSaveDialog = false;
    } else if (ch == '\n' || ch == KEY_ENTER) {
        saveC3D();
        mShowSaveDialog = false;
        mModified = false;  // Reset modified flag after save
        // Exit edit mode after save (from exit confirm flow)
        if (mEditMode) {
            mEditMode = false;
            mFirstMarker = -1;
            mSecondMarker = -1;
        }
    } else if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
        if (!mSaveFilename.empty()) {
            mSaveFilename.pop_back();
        }
    } else if (ch >= 32 && ch < 127) {  // Printable ASCII
        if (mSaveFilename.length() < 44) {
            mSaveFilename += static_cast<char>(ch);
        }
    }
}

void C3DInspectorUI::drawExitConfirm() {
    int maxY, maxX;
    getmaxyx(stdscr, maxY, maxX);

    int dialogWidth = 40;
    int dialogHeight = 5;
    int startX = (maxX - dialogWidth) / 2;
    int startY = (maxY - dialogHeight) / 2;

    // Draw dialog box
    for (int y = startY; y < startY + dialogHeight; ++y) {
        mvhline(y, startX, ' ', dialogWidth);
    }

    // Border
    mvhline(startY, startX, '-', dialogWidth);
    mvhline(startY + dialogHeight - 1, startX, '-', dialogWidth);
    for (int y = startY; y < startY + dialogHeight; ++y) {
        mvaddch(y, startX, '|');
        mvaddch(y, startX + dialogWidth - 1, '|');
    }

    // Content
    attron(A_BOLD);
    mvprintw(startY + 1, startX + 2, "Save changes before exit?");
    attroff(A_BOLD);

    mvprintw(startY + 3, startX + 2, "[y] Save  [n] Discard  [c] Cancel");
}

void C3DInspectorUI::handleExitConfirmInput(int ch) {
    if (ch == 'y' || ch == 'Y') {
        // Save and exit edit mode
        mSaveFilename = getDefaultSaveFilename();
        mShowExitConfirm = false;
        mShowSaveDialog = true;
    } else if (ch == 'n' || ch == 'N') {
        // Discard and exit edit mode
        mShowExitConfirm = false;
        mEditMode = false;
        mFirstMarker = -1;
        mSecondMarker = -1;
        // Reload to discard changes
        loadC3DInfo();
    } else if (ch == 'c' || ch == 'C' || ch == 27) {
        // Cancel - stay in edit mode
        mShowExitConfirm = false;
    }
}

void C3DInspectorUI::loadDirectory() {
    mDirEntries.clear();
    mC3DFiles.clear();

    if (mCurrentDir.empty()) {
        mCurrentDir = "data/motion";
    }

    try {
        if (!fs::exists(mCurrentDir) || !fs::is_directory(mCurrentDir)) {
            return;
        }

        for (const auto& entry : fs::directory_iterator(mCurrentDir)) {
            std::string name = entry.path().filename().string();
            if (entry.is_directory()) {
                mDirEntries.push_back(name + "/");
            } else if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".c3d") {
                    mC3DFiles.push_back(name);
                }
            }
        }

        std::sort(mDirEntries.begin(), mDirEntries.end());
        std::sort(mC3DFiles.begin(), mC3DFiles.end());

    } catch (const fs::filesystem_error&) {
        // Directory access error
    }
}

void C3DInspectorUI::loadC3DFromPath(const std::string& path) {
    mC3DInfo = C3DInfo();
    mC3D.reset();
    mEditMode = false;
    mCursorPos = 0;
    mFirstMarker = -1;
    mSecondMarker = -1;
    mModified = false;
    mShowSaveDialog = false;
    mShowExitConfirm = false;

    // Reset NaN inspection state
    mNanInspectMode = false;
    mNanMarkerCursor = 0;
    mNanMarkerScrollOffset = 0;
    mNanMarkerIndices.clear();
    mSelectedNanMarker = -1;
    mNanFrameList.clear();
    mNanFrameCursor = 0;
    mNanFrameScrollOffset = 0;
    mNanShowFrameDetail = false;
    mInterpolatedFrames.clear();

    try {
        mLocalPath = path;
        mC3DInfo.uri = path;
        mC3DInfo.filename = fs::path(path).filename().string();
        mC3D = std::make_unique<ezc3d::c3d>(path);

        mC3DInfo.numFrames = mC3D->data().nbFrames();
        mC3DInfo.frameRate = mC3D->header().frameRate();
        mC3DInfo.duration = mC3DInfo.numFrames / mC3DInfo.frameRate;

        try {
            mC3DInfo.labels = mC3D->parameters().group("POINT").parameter("LABELS").valuesAsString();
        } catch (...) {}

        // Count NaN frames per marker
        size_t numMarkers = mC3DInfo.labels.size();
        mC3DInfo.nanCounts.resize(numMarkers, 0);
        for (size_t f = 0; f < mC3DInfo.numFrames; ++f) {
            const auto& points = mC3D->data().frame(f).points();
            for (size_t m = 0; m < numMarkers; ++m) {
                const auto& p = points.point(m);
                if (std::isnan(p.x()) || std::isnan(p.y()) || std::isnan(p.z())) {
                    mC3DInfo.nanCounts[m]++;
                }
            }
        }

        mC3DInfo.loaded = true;
    } catch (...) {
        mC3DInfo.loaded = false;
    }
}

void C3DInspectorUI::drawDirBrowse() {
    int maxY, maxX;
    getmaxyx(stdscr, maxY, maxX);

    // Header
    attron(A_BOLD);
    mvprintw(0, 0, "C3D Inspector - Browse: %s", mCurrentDir.c_str());
    attroff(A_BOLD);
    mvhline(1, 0, '-', maxX);

    int maxVisible = getMaxVisibleItems();
    int totalItems = (int)mDirEntries.size() + (int)mC3DFiles.size();

    if (totalItems == 0) {
        mvprintw(3, 2, "(empty directory or no .c3d files)");
    } else {
        int startIdx = mDirScrollOffset;
        int endIdx = std::min(startIdx + maxVisible, totalItems);

        int row = 3;
        for (int i = startIdx; i < endIdx; ++i) {
            bool isDir = i < (int)mDirEntries.size();
            const std::string& name = isDir ? mDirEntries[i] : mC3DFiles[i - mDirEntries.size()];

            if (i == mSelectedDirEntry) {
                attron(A_REVERSE);
                mvprintw(row, 2, "%-60s", name.c_str());
                attroff(A_REVERSE);
            } else {
                if (isDir) {
                    attron(A_BOLD);
                    mvprintw(row, 2, "%-60s", name.c_str());
                    attroff(A_BOLD);
                } else {
                    mvprintw(row, 2, "%-60s", name.c_str());
                }
            }
            row++;
        }
    }

    // Footer
    mvhline(maxY - 2, 0, '-', maxX);
    mvprintw(maxY - 1, 0, "[UP/DOWN] Navigate  [ENTER] Select  [BACKSPACE] Parent dir  [q] Quit  (%d/%d)",
             totalItems > 0 ? mSelectedDirEntry + 1 : 0, totalItems);
}

void C3DInspectorUI::handleInput(int ch) {
    int maxVisible = getMaxVisibleItems();

    switch (mStage) {
    case PID_SELECT:
        if (ch == KEY_UP && mSelectedPID > 0) {
            mSelectedPID--;
            if (mSelectedPID < mScrollOffset) mScrollOffset = mSelectedPID;
        } else if (ch == KEY_DOWN && mSelectedPID < (int)mPIDs.size() - 1) {
            mSelectedPID++;
            if (mSelectedPID >= mScrollOffset + maxVisible) mScrollOffset = mSelectedPID - maxVisible + 1;
        } else if (ch == '\n' || ch == KEY_ENTER) {
            mStage = PREPOST_SELECT;
            mSelectedPrePost = 0;
        } else if (ch == 'd' || ch == 'D') {
            // Enter directory browsing mode
            mCurrentDir = "data/motion";
            loadDirectory();
            mSelectedDirEntry = 0;
            mDirScrollOffset = 0;
            mStage = DIR_BROWSE;
        } else if (ch == 'm' || ch == 'M') {
            // Batch merge all PIDs
            runBatchMerge();
        }
        break;

    case PREPOST_SELECT:
        if (ch == KEY_UP && mSelectedPrePost > 0) {
            mSelectedPrePost--;
        } else if (ch == KEY_DOWN && mSelectedPrePost < 1) {
            mSelectedPrePost++;
        } else if (ch == '\n' || ch == KEY_ENTER) {
            loadFiles();
            mSelectedFile = 0;
            mScrollOffset = 0;
            mStage = FILE_SELECT;
        } else if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
            mStage = PID_SELECT;
        }
        break;

    case FILE_SELECT:
        if (ch == KEY_UP && mSelectedFile > 0) {
            mSelectedFile--;
            if (mSelectedFile < mScrollOffset) mScrollOffset = mSelectedFile;
        } else if (ch == KEY_DOWN && mSelectedFile < (int)mFiles.size() - 1) {
            mSelectedFile++;
            if (mSelectedFile >= mScrollOffset + maxVisible) mScrollOffset = mSelectedFile - maxVisible + 1;
        } else if ((ch == '\n' || ch == KEY_ENTER) && !mFiles.empty()) {
            loadC3DInfo();
            mScrollOffset = 0;
            mStage = INSPECT_VIEW;
        } else if (ch == 'c' || ch == 'C') {
            // Enter concatenation mode
            mConcatFilter = "trimmed";  // Default filter
            applyFileFilter();           // Populate mFilteredIndices
            mShowConcatDialog = true;
        } else if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
            mStage = PREPOST_SELECT;
        }
        break;

    case INSPECT_VIEW: {
        int maxY, maxX;
        getmaxyx(stdscr, maxY, maxX);
        (void)maxX;

        // Handle NaN inspection mode
        if (mNanInspectMode) {
            handleNanInspectInput(ch);
            break;
        }

        if (mEditMode) {
            // Edit mode navigation and selection
            int numLabels = (int)mC3DInfo.labels.size();
            int maxVisibleLabels = maxY - 14;  // Same as in drawInspectView

            if (ch == KEY_UP && mCursorPos > 0) {
                mCursorPos--;
                // Adjust scroll to keep cursor visible
                if (mCursorPos < mScrollOffset) {
                    mScrollOffset = mCursorPos;
                }
            } else if (ch == KEY_DOWN && mCursorPos < numLabels - 1) {
                mCursorPos++;
                // Adjust scroll to keep cursor visible
                if (mCursorPos >= mScrollOffset + maxVisibleLabels) {
                    mScrollOffset = mCursorPos - maxVisibleLabels + 1;
                }
            } else if (ch == '\n' || ch == KEY_ENTER) {
                // Select marker
                if (mFirstMarker < 0) {
                    mFirstMarker = mCursorPos;
                } else if (mSecondMarker < 0 && mCursorPos != mFirstMarker) {
                    mSecondMarker = mCursorPos;
                }
            } else if (ch == 'y' || ch == 'Y') {
                // Confirm swap
                if (mFirstMarker >= 0 && mSecondMarker >= 0) {
                    swapMarkerData(mFirstMarker, mSecondMarker);
                    mFirstMarker = -1;
                    mSecondMarker = -1;
                }
            } else if (ch == 'n' || ch == 'N') {
                // Cancel selection
                mFirstMarker = -1;
                mSecondMarker = -1;
            } else if (ch == 27) {  // ESC
                if (mModified) {
                    mShowExitConfirm = true;
                } else {
                    mEditMode = false;
                    mFirstMarker = -1;
                    mSecondMarker = -1;
                }
            }
        } else {
            // Normal view mode
            if (ch == KEY_UP && mScrollOffset > 0) {
                mScrollOffset--;
            } else if (ch == KEY_DOWN && mScrollOffset < (int)mC3DInfo.labels.size() - 1) {
                mScrollOffset++;
            } else if (ch == 'e' || ch == 'E') {
                // Enter edit mode
                mEditMode = true;
                mCursorPos = mScrollOffset;  // Start cursor at current scroll position
                mFirstMarker = -1;
                mSecondMarker = -1;
            } else if (ch == 'n' || ch == 'N') {
                // Enter NaN inspection mode
                buildNanMarkerList();
                mNanMarkerCursor = 0;
                mNanMarkerScrollOffset = 0;
                mSelectedNanMarker = -1;
                mNanShowFrameDetail = false;
                mNanInspectMode = true;
            } else if ((ch == 's' || ch == 'S') && mModified) {
                // Open save dialog
                mSaveFilename = getDefaultSaveFilename();
                mShowSaveDialog = true;
            } else if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
                mScrollOffset = 0;
                mStage = FILE_SELECT;
            }
        }
        break;
    }  // end INSPECT_VIEW

    case DIR_BROWSE: {
        int totalItems = (int)mDirEntries.size() + (int)mC3DFiles.size();

        if (ch == KEY_UP && mSelectedDirEntry > 0) {
            mSelectedDirEntry--;
            if (mSelectedDirEntry < mDirScrollOffset) {
                mDirScrollOffset = mSelectedDirEntry;
            }
        } else if (ch == KEY_DOWN && mSelectedDirEntry < totalItems - 1) {
            mSelectedDirEntry++;
            if (mSelectedDirEntry >= mDirScrollOffset + maxVisible) {
                mDirScrollOffset = mSelectedDirEntry - maxVisible + 1;
            }
        } else if (ch == '\n' || ch == KEY_ENTER) {
            if (totalItems > 0) {
                bool isDir = mSelectedDirEntry < (int)mDirEntries.size();
                if (isDir) {
                    // Enter subdirectory
                    std::string dirName = mDirEntries[mSelectedDirEntry];
                    // Remove trailing '/'
                    if (!dirName.empty() && dirName.back() == '/') {
                        dirName.pop_back();
                    }
                    mCurrentDir = mCurrentDir + "/" + dirName;
                    loadDirectory();
                    mSelectedDirEntry = 0;
                    mDirScrollOffset = 0;
                } else {
                    // Open C3D file
                    std::string fileName = mC3DFiles[mSelectedDirEntry - mDirEntries.size()];
                    std::string fullPath = mCurrentDir + "/" + fileName;
                    loadC3DFromPath(fullPath);
                    mScrollOffset = 0;
                    mStage = INSPECT_VIEW;
                }
            }
        } else if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
            // Go to parent directory
            fs::path p(mCurrentDir);
            fs::path parent = p.parent_path();
            if (!parent.empty() && parent != p) {
                mCurrentDir = parent.string();
                loadDirectory();
                mSelectedDirEntry = 0;
                mDirScrollOffset = 0;
            } else {
                // At root, go back to PID select
                mStage = PID_SELECT;
            }
        }
        break;
    }  // end DIR_BROWSE

    }  // end switch
}

void C3DInspectorUI::run() {
    setlocale(LC_ALL, "");  // Enable UTF-8 support
    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0);

    // Initialize colors for NaN display
    start_color();
    init_pair(1, COLOR_GREEN, COLOR_BLACK);  // Green for markers without NaN
    init_pair(2, COLOR_RED, COLOR_BLACK);    // Red for NaN markers

    loadPIDs();

    bool running = true;
    while (running) {
        clear();

        switch (mStage) {
        case PID_SELECT:
            drawPIDSelect();
            break;
        case PREPOST_SELECT:
            drawPrePostSelect();
            break;
        case FILE_SELECT:
            drawFileSelect();
            break;
        case INSPECT_VIEW:
            if (mNanInspectMode) {
                drawNanInspectView();
            } else {
                drawInspectView();
            }
            break;
        case DIR_BROWSE:
            drawDirBrowse();
            break;
        }

        // Draw dialogs on top if active
        if (mShowExitConfirm) {
            drawExitConfirm();
        } else if (mShowSaveDialog) {
            drawSaveDialog();
        } else if (mShowConcatDialog) {
            drawConcatDialog();
        }

        refresh();

        int ch = getch();
        if (ch == 'q' || ch == 'Q') {
            if (!mShowSaveDialog && !mShowExitConfirm && !mShowConcatDialog && !mEditMode && !mNanInspectMode) {
                running = false;
            }
        } else if (mShowExitConfirm) {
            handleExitConfirmInput(ch);
        } else if (mShowSaveDialog) {
            handleSaveDialogInput(ch);
        } else if (mShowConcatDialog) {
            handleConcatDialogInput(ch);
        } else {
            handleInput(ch);
        }
    }

    endwin();
}

// Non-interactive inspection for command-line use
void inspectC3D(rm::ResourceManager& mgr, const std::string& uri) {
    std::cout << "URI: " << uri << "\n";
    std::cout << std::string(60, '-') << "\n";

    try {
        auto handle = mgr.fetch(uri);
        ezc3d::c3d c3d(handle.local_path().string());

        size_t numFrames = c3d.data().nbFrames();
        double frameRate = c3d.header().frameRate();
        double duration = numFrames / frameRate;

        std::cout << "Total Frames: " << numFrames << "\n";
        std::cout << "Frame Rate:   " << frameRate << " Hz\n";
        std::cout << "Duration:     " << std::fixed << std::setprecision(2) << duration << " s\n";
        std::cout << "\n";

        std::vector<std::string> labels;
        try {
            labels = c3d.parameters().group("POINT").parameter("LABELS").valuesAsString();
        } catch (...) {
            std::cout << "No POINT/LABELS found in file\n";
            return;
        }

        std::cout << "Labels (" << labels.size() << " markers):\n";
        std::cout << std::string(60, '-') << "\n";

        for (size_t i = 0; i < labels.size(); ++i) {
            std::cout << std::setw(4) << i << ": " << labels[i] << "\n";
        }

    } catch (const rm::RMError& e) {
        std::cerr << "Resource error: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

void printUsage(const char* progName) {
    std::cerr << "Usage: " << progName << " [options] [uri]\n\n";
    std::cerr << "Interactive mode (ncurses):\n";
    std::cerr << "  " << progName << "                      # interactive PID/file selection\n\n";
    std::cerr << "Direct inspection:\n";
    std::cerr << "  " << progName << " @pid:CP001/pre/gait/file.c3d\n";
    std::cerr << "  " << progName << " /path/to/file.c3d\n\n";
    std::cerr << "Batch operations:\n";
    std::cerr << "  " << progName << " --batch-merge        # merge trimmed_* files for all PIDs\n";
}

// Non-interactive batch merge for all PIDs
void runBatchMergeNonInteractive(rm::ResourceManager& mgr) {
    std::string filterLower = "trimmed_";

    // Get all PIDs
    std::vector<std::string> pids;
    try {
        pids = mgr.list("@pid:");
        std::sort(pids.begin(), pids.end());
    } catch (const rm::RMError& e) {
        std::cerr << "Error listing PIDs: " << e.what() << std::endl;
        return;
    }

    std::cout << "Found " << pids.size() << " PIDs" << std::endl;

    int total = pids.size() * 3;  // pre, op1, op2
    int current = 0;
    int merged = 0;

    for (const auto& pid : pids) {
        for (const auto& visit : {"pre", "op1", "op2"}) {
            current++;
            std::string uri = "@pid:" + pid + "/" + visit + "/gait";

            // List and filter files
            std::vector<std::string> trimmedFiles;
            try {
                auto files = mgr.list(uri);
                for (const auto& f : files) {
                    std::string fLower = f;
                    std::transform(fLower.begin(), fLower.end(), fLower.begin(), ::tolower);

                    if (fLower.find(filterLower) == 0 &&
                        fLower.find("_unified.c3d") == std::string::npos) {
                        trimmedFiles.push_back(f);
                    }
                }
            } catch (const rm::RMError&) {
                continue;  // Directory doesn't exist
            }

            if (trimmedFiles.empty()) {
                continue;
            }

            std::sort(trimmedFiles.begin(), trimmedFiles.end());

            std::cout << "[" << current << "/" << total << "] " << pid << "/" << visit
                      << ": merging " << trimmedFiles.size() << " files..." << std::flush;

            try {
                // Load first file to get labels and frame rate
                auto firstHandle = mgr.fetch(uri + "/" + trimmedFiles[0]);
                ezc3d::c3d firstC3d(firstHandle.local_path().string());

                double frameRate = firstC3d.header().frameRate();
                std::vector<std::string> labels;
                try {
                    labels = firstC3d.parameters().group("POINT").parameter("LABELS").valuesAsString();
                } catch (...) {}

                // Create output c3d
                ezc3d::c3d output;

                ezc3d::ParametersNS::GroupNS::Parameter rateParam("RATE");
                rateParam.set(frameRate);
                output.parameter("POINT", rateParam);

                if (!labels.empty()) {
                    output.point(labels);
                }

                // Merge all frames
                size_t totalFrames = 0;
                for (const auto& filename : trimmedFiles) {
                    auto handle = mgr.fetch(uri + "/" + filename);
                    ezc3d::c3d inputC3d(handle.local_path().string());

                    size_t numFrames = inputC3d.data().nbFrames();
                    totalFrames += numFrames;
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
                }

                // Determine output path
                std::string outputName = "trimmed_unified.c3d";
                fs::path outputPath;

                if (mgr.exists(uri + "/" + outputName)) {
                    auto outputHandle = mgr.fetch(uri + "/" + outputName);
                    outputPath = outputHandle.local_path();
                } else {
                    fs::path c3dDir = firstHandle.local_path().parent_path();
                    if (c3dDir.filename() == "Generated_C3D_files") {
                        outputPath = c3dDir.parent_path() / outputName;
                    } else {
                        outputPath = c3dDir / outputName;
                    }
                }

                output.write(outputPath.string());
                std::cout << " OK (" << totalFrames << " frames)" << std::endl;
                merged++;

            } catch (const std::exception& e) {
                std::cout << " FAILED: " << e.what() << std::endl;
            }
        }
    }

    std::cout << "\nBatch merge complete: " << merged << " unified files created" << std::endl;
}

int main(int argc, char* argv[]) {
    auto& mgr = rm::getManager();

    if (argc < 2) {
        // Interactive mode
        C3DInspectorUI ui(mgr);
        ui.run();
        return 0;
    }

    std::string arg = argv[1];

    if (arg == "-h" || arg == "--help") {
        printUsage(argv[0]);
        return 0;
    }

    if (arg == "--batch-merge") {
        // Non-interactive batch merge
        runBatchMergeNonInteractive(mgr);
        return 0;
    }

    // Direct inspection mode
    if (arg.find(".c3d") != std::string::npos || arg.find(".C3D") != std::string::npos) {
        inspectC3D(mgr, arg);
    } else {
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}
