#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <cmath>
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
        mSelectedDirEntry(0), mDirScrollOffset(0) {}

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

    std::string prePost = (mSelectedPrePost == 0) ? "pre" : "post";
    std::string uri = "@pid:" + mPIDs[mSelectedPID] + "/gait/" + prePost;

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

    if (mFiles.empty()) return;

    std::string prePost = (mSelectedPrePost == 0) ? "pre" : "post";
    mC3DInfo.uri = "@pid:" + mPIDs[mSelectedPID] + "/gait/" + prePost + "/" + mFiles[mSelectedFile];
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
    mvprintw(maxY - 1, 0, "[UP/DOWN] Navigate  [ENTER] Select  [d] Browse data/motion  [q] Quit  (%d/%d)",
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
    mvprintw(maxY - 1, 0, "[UP/DOWN] Navigate  [ENTER] Select  [BACKSPACE] Back  [q] Quit  (%d/%d)",
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
            mvprintw(maxY - 1, 0, "[UP/DOWN] Scroll  [e] Edit  [s] Save  [BACKSPACE] Back  [q] Quit  (%d/%d)",
                     mC3DInfo.labels.empty() ? 0 : mScrollOffset + 1, (int)mC3DInfo.labels.size());
        } else {
            mvprintw(maxY - 1, 0, "[UP/DOWN] Scroll  [e] Edit  [BACKSPACE] Back  [q] Quit  (%d/%d)",
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
    return filename + "_fix.c3d";
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
        } else if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
            mStage = PREPOST_SELECT;
        }
        break;

    case INSPECT_VIEW: {
        int maxY, maxX;
        getmaxyx(stdscr, maxY, maxX);
        (void)maxX;

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
            drawInspectView();
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
        }

        refresh();

        int ch = getch();
        if (ch == 'q' || ch == 'Q') {
            if (!mShowSaveDialog && !mShowExitConfirm && !mEditMode) {
                running = false;
            }
        } else if (mShowExitConfirm) {
            handleExitConfirmInput(ch);
        } else if (mShowSaveDialog) {
            handleSaveDialogInput(ch);
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
    std::cerr << "Usage: " << progName << " [uri]\n\n";
    std::cerr << "Interactive mode (ncurses):\n";
    std::cerr << "  " << progName << "                      # interactive PID/file selection\n\n";
    std::cerr << "Direct inspection:\n";
    std::cerr << "  " << progName << " @pid:CP001/gait/pre/file.c3d\n";
    std::cerr << "  " << progName << " /path/to/file.c3d\n";
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

    // Direct inspection mode
    if (arg.find(".c3d") != std::string::npos || arg.find(".C3D") != std::string::npos) {
        inspectC3D(mgr, arg);
    } else {
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}
