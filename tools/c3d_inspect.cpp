#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <locale.h>
#include <ncurses.h>
#include <ezc3d/ezc3d_all.h>
#include "rm/rm.hpp"

// UI State
enum Stage { PID_SELECT, PREPOST_SELECT, FILE_SELECT, INSPECT_VIEW };

struct C3DInfo {
    std::string uri;
    std::string filename;
    size_t numFrames;
    double frameRate;
    double duration;
    std::vector<std::string> labels;
    bool loaded;

    C3DInfo() : numFrames(0), frameRate(0), duration(0), loaded(false) {}
};

class C3DInspectorUI {
public:
    C3DInspectorUI(rm::ResourceManager& mgr) : mMgr(mgr), mStage(PID_SELECT),
        mSelectedPID(0), mSelectedPrePost(0), mSelectedFile(0), mScrollOffset(0) {}

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

    void loadPIDs();
    void loadPIDMetadata();
    void loadFiles();
    void loadC3DInfo();

    void drawPIDSelect();
    void drawPrePostSelect();
    void drawFileSelect();
    void drawInspectView();

    void handleInput(int ch);
    int getMaxVisibleItems();
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
    if (mFiles.empty()) return;

    std::string prePost = (mSelectedPrePost == 0) ? "pre" : "post";
    mC3DInfo.uri = "@pid:" + mPIDs[mSelectedPID] + "/gait/" + prePost + "/" + mFiles[mSelectedFile];
    mC3DInfo.filename = mFiles[mSelectedFile];

    try {
        auto handle = mMgr.fetch(mC3DInfo.uri);
        ezc3d::c3d c3d(handle.local_path().string());

        mC3DInfo.numFrames = c3d.data().nbFrames();
        mC3DInfo.frameRate = c3d.header().frameRate();
        mC3DInfo.duration = mC3DInfo.numFrames / mC3DInfo.frameRate;

        try {
            mC3DInfo.labels = c3d.parameters().group("POINT").parameter("LABELS").valuesAsString();
        } catch (...) {}

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
    mvprintw(maxY - 1, 0, "[UP/DOWN] Navigate  [ENTER] Select  [q] Quit  (%d/%d)",
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

    attron(A_BOLD);
    mvprintw(0, 0, "C3D Inspector - %s", mC3DInfo.filename.c_str());
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
            mvprintw(row, 2, "%4d: %s", i, mC3DInfo.labels[i].c_str());
        }
    }

    mvhline(maxY - 2, 0, '-', maxX);
    mvprintw(maxY - 1, 0, "[UP/DOWN] Scroll labels  [BACKSPACE] Back  [q] Quit  (%d/%d)",
             mC3DInfo.labels.empty() ? 0 : mScrollOffset + 1, (int)mC3DInfo.labels.size());
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

    case INSPECT_VIEW:
        if (ch == KEY_UP && mScrollOffset > 0) {
            mScrollOffset--;
        } else if (ch == KEY_DOWN && mScrollOffset < (int)mC3DInfo.labels.size() - 1) {
            mScrollOffset++;
        } else if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
            mScrollOffset = 0;
            mStage = FILE_SELECT;
        }
        break;
    }
}

void C3DInspectorUI::run() {
    setlocale(LC_ALL, "");  // Enable UTF-8 support
    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0);

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
        }

        refresh();

        int ch = getch();
        if (ch == 'q' || ch == 'Q') {
            running = false;
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
    rm::ResourceManager mgr("data/rm_config.yaml");

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
