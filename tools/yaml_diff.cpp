// yaml_diff.cpp - YAML Config Diff Viewer (ncurses TUI)
// Compare multiple YAML configuration files side-by-side with flattened key-path diff.

#include <ncurses.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

// ─── Stage enum ─────────────────────────────────────────────────────────────

enum Stage { FILE_LIST, DIFF_VIEW };

// ─── Data structures ────────────────────────────────────────────────────────

struct YamlFileEntry {
    std::string relativePath;
    std::string fullPath;
    bool selected = false;
};

struct DiffRow {
    std::string keyPath;
    std::vector<std::string> values; // one per selected file
    bool allSame = true;
    bool anyAbsent = false;
};

// ─── Color pairs ────────────────────────────────────────────────────────────

enum ColorPair {
    CP_SAME      = 1, // GREEN/BLACK  - identical values
    CP_DIFFER    = 2, // RED/BLACK    - values differ
    CP_ABSENT    = 3, // YELLOW/BLACK - key absent from some file
    CP_HEADER    = 4, // CYAN/BLACK   - headers, separators
    CP_SELECTED  = 5, // WHITE/BLUE   - selected file highlight
    CP_BG        = 6, // WHITE/BLACK  - default dark background
    CP_POPUP     = 7, // WHITE/BLACK  - popup window
    CP_POPUP_HDR = 8, // CYAN/BLACK   - popup header
};

// ─── YamlDiffUI ─────────────────────────────────────────────────────────────

class YamlDiffUI {
public:
    explicit YamlDiffUI(const std::string& dir) : mDir(dir) {}

    void run() {
        setlocale(LC_ALL, "");
        initscr();
        cbreak();
        noecho();
        keypad(stdscr, TRUE);
        curs_set(0);

        start_color();
        use_default_colors();

        if (COLORS >= 256) {
            // Index 16 = true #000000, immune to terminal theme remapping
            const short BG = 16;
            init_pair(CP_SAME,      47,  BG);  // bright green
            init_pair(CP_DIFFER,    204, BG);  // bright red/pink
            init_pair(CP_ABSENT,    226, BG);  // vivid yellow
            init_pair(CP_HEADER,    81,  BG);  // bright cyan
            init_pair(CP_SELECTED,  BG,  255); // black on bright white
            init_pair(CP_BG,        252, BG);  // light grey text
            init_pair(CP_POPUP,     252, 235); // light grey on dark grey
            init_pair(CP_POPUP_HDR, 81,  235); // cyan on dark grey
        } else {
            init_pair(CP_SAME,      COLOR_GREEN,  COLOR_BLACK);
            init_pair(CP_DIFFER,    COLOR_RED,    COLOR_BLACK);
            init_pair(CP_ABSENT,    COLOR_YELLOW, COLOR_BLACK);
            init_pair(CP_HEADER,    COLOR_CYAN,   COLOR_BLACK);
            init_pair(CP_SELECTED,  COLOR_BLACK,  COLOR_WHITE);
            init_pair(CP_BG,        COLOR_WHITE,  COLOR_BLACK);
            init_pair(CP_POPUP,     COLOR_WHITE,  COLOR_BLACK);
            init_pair(CP_POPUP_HDR, COLOR_CYAN,   COLOR_BLACK);
        }

        bkgdset(COLOR_PAIR(CP_BG));
        erase();

        scanFiles();
        applyFilter();

        while (true) {
            erase();
            if (mStage == FILE_LIST) {
                drawFileList();
            } else {
                drawDiffView();
            }
            refresh();

            int ch = getch();
            if (ch == 'q' || ch == 'Q') break;

            if (mStage == FILE_LIST)
                handleFileListInput(ch);
            else
                handleDiffViewInput(ch);
        }
        endwin();
    }

private:
    // ─── File scanning ──────────────────────────────────────────────────

    void scanFiles() {
        mAllFiles.clear();
        if (!fs::is_directory(mDir)) return;

        std::vector<std::string> paths;
        for (auto& entry : fs::recursive_directory_iterator(mDir)) {
            if (entry.is_regular_file()) {
                auto ext = entry.path().extension().string();
                if (ext == ".yaml" || ext == ".yml") {
                    paths.push_back(entry.path().string());
                }
            }
        }
        std::sort(paths.begin(), paths.end());

        for (auto& p : paths) {
            YamlFileEntry e;
            e.fullPath = p;
            e.relativePath = fs::relative(fs::path(p), fs::path(mDir)).string();
            mAllFiles.push_back(e);
        }
    }

    void ensureContentCache() {
        if (mContentCached) return;
        mContentCache.resize(mAllFiles.size());
        for (size_t i = 0; i < mAllFiles.size(); ++i) {
            std::ifstream ifs(mAllFiles[i].fullPath);
            if (ifs) {
                std::string content((std::istreambuf_iterator<char>(ifs)),
                                     std::istreambuf_iterator<char>());
                mContentCache[i] = toLower(content);
            }
        }
        mContentCached = true;
    }

    bool fileContentMatches(size_t fileIdx, const std::string& lowerQuery) {
        if (lowerQuery.empty()) return true;
        ensureContentCache();
        return mContentCache[fileIdx].find(lowerQuery) != std::string::npos;
    }

    void applyFilter() {
        mFiltered.clear();
        std::string lowerFilter = toLower(mFilterStr);
        std::string lowerGrep = toLower(mGrepStr);
        for (size_t i = 0; i < mAllFiles.size(); ++i) {
            // filename filter
            if (!lowerFilter.empty() &&
                toLower(mAllFiles[i].relativePath).find(lowerFilter) == std::string::npos)
                continue;
            // content grep filter
            if (!lowerGrep.empty() && !fileContentMatches(i, lowerGrep))
                continue;
            mFiltered.push_back(i);
        }
        // clamp cursor
        if (mCursor >= (int)mFiltered.size()) mCursor = std::max(0, (int)mFiltered.size() - 1);
        if (mScrollOffset > mCursor) mScrollOffset = mCursor;
    }

    // ─── YAML flattening ────────────────────────────────────────────────

    void flattenYaml(const YAML::Node& node, const std::string& prefix,
                     std::map<std::string, std::string>& out) {
        if (node.IsMap()) {
            for (auto it = node.begin(); it != node.end(); ++it) {
                std::string key = it->first.as<std::string>();
                std::string path = prefix.empty() ? key : prefix + "." + key;
                flattenYaml(it->second, path, out);
            }
        } else if (node.IsSequence()) {
            for (size_t i = 0; i < node.size(); ++i) {
                std::string path = prefix + "[" + std::to_string(i) + "]";
                flattenYaml(node[i], path, out);
            }
        } else if (node.IsScalar()) {
            out[prefix] = node.as<std::string>();
        } else if (node.IsNull()) {
            out[prefix] = "~";
        }
    }

    // ─── Diff computation ───────────────────────────────────────────────

    void loadSelectedFiles() {
        mSelectedNames.clear();
        mFlatMaps.clear();
        for (auto& f : mAllFiles) {
            if (!f.selected) continue;
            mSelectedNames.push_back(f.relativePath);
            std::map<std::string, std::string> flat;
            try {
                YAML::Node root = YAML::LoadFile(f.fullPath);
                flattenYaml(root, "", flat);
            } catch (...) {
                // leave flat empty on parse error
            }
            mFlatMaps.push_back(flat);
        }
    }

    void computeDiff() {
        mDiffRows.clear();
        mVisibleRows.clear();

        // union all keys
        std::set<std::string> allKeys;
        for (auto& m : mFlatMaps)
            for (auto& kv : m)
                allKeys.insert(kv.first);

        size_t nFiles = mFlatMaps.size();
        for (auto& key : allKeys) {
            DiffRow row;
            row.keyPath = key;
            row.values.resize(nFiles);
            row.allSame = true;
            row.anyAbsent = false;

            std::string firstVal;
            bool hasFirst = false;

            for (size_t i = 0; i < nFiles; ++i) {
                auto it = mFlatMaps[i].find(key);
                if (it == mFlatMaps[i].end()) {
                    row.values[i] = "[absent]";
                    row.anyAbsent = true;
                    row.allSame = false;
                } else {
                    row.values[i] = it->second;
                    if (!hasFirst) {
                        firstVal = it->second;
                        hasFirst = true;
                    } else if (it->second != firstVal) {
                        row.allSame = false;
                    }
                }
            }
            mDiffRows.push_back(row);
        }
        updateVisibleRows();
    }

    void updateVisibleRows() {
        mVisibleRows.clear();
        for (size_t i = 0; i < mDiffRows.size(); ++i) {
            if (!mDiffsOnly || !mDiffRows[i].allSame)
                mVisibleRows.push_back(i);
        }
        if (mDiffScroll >= (int)mVisibleRows.size())
            mDiffScroll = std::max(0, (int)mVisibleRows.size() - 1);
    }

    // ─── Drawing: File List ─────────────────────────────────────────────

    void drawFileList() {
        int maxY, maxX;
        getmaxyx(stdscr, maxY, maxX);

        int selectedCount = 0;
        for (auto& f : mAllFiles) if (f.selected) selectedCount++;

        // Title
        attron(COLOR_PAIR(CP_HEADER) | A_BOLD);
        std::string title = " YAML Diff - " + mDir + " (" + std::to_string(mAllFiles.size()) + " files)";
        mvprintw(0, 0, "%-*s", maxX, title.c_str());
        attroff(COLOR_PAIR(CP_HEADER) | A_BOLD);

        // Filter line (filename)
        if (mFilterMode) {
            attron(A_BOLD);
            mvprintw(1, 0, " Filter: %s_", mFilterStr.c_str());
            attroff(A_BOLD);
        } else if (!mFilterStr.empty()) {
            mvprintw(1, 0, " Filter: %s", mFilterStr.c_str());
        } else {
            mvprintw(1, 0, " Filter: (none)");
        }

        // Grep line (content)
        if (mGrepMode) {
            attron(COLOR_PAIR(CP_ABSENT) | A_BOLD);
            mvprintw(2, 0, "   Grep: %s_", mGrepStr.c_str());
            attroff(COLOR_PAIR(CP_ABSENT) | A_BOLD);
        } else if (!mGrepStr.empty()) {
            attron(COLOR_PAIR(CP_ABSENT));
            mvprintw(2, 0, "   Grep: %s (%d hits)", mGrepStr.c_str(), (int)mFiltered.size());
            attroff(COLOR_PAIR(CP_ABSENT));
        }

        // Separator
        attron(COLOR_PAIR(CP_HEADER));
        mvhline(3, 0, ACS_HLINE, maxX);
        attroff(COLOR_PAIR(CP_HEADER));

        // File list
        int listStart = 4;
        int listHeight = maxY - listStart - 2; // reserve bottom status bar
        if (listHeight < 1) listHeight = 1;

        // ensure cursor visible
        if (mCursor < mScrollOffset) mScrollOffset = mCursor;
        if (mCursor >= mScrollOffset + listHeight) mScrollOffset = mCursor - listHeight + 1;

        for (int i = 0; i < listHeight; ++i) {
            int idx = mScrollOffset + i;
            if (idx >= (int)mFiltered.size()) break;

            int fileIdx = mFiltered[idx];
            auto& f = mAllFiles[fileIdx];
            bool isCursor = (idx == mCursor);

            std::string marker = f.selected ? "[*]" : "[ ]";
            std::string line = "  " + marker + " " + f.relativePath;

            if (isCursor) attron(A_REVERSE);
            if (f.selected) attron(COLOR_PAIR(CP_SELECTED));

            mvprintw(listStart + i, 0, "%-*s", maxX, line.c_str());

            if (f.selected) attroff(COLOR_PAIR(CP_SELECTED));
            if (isCursor) attroff(A_REVERSE);
        }

        // Status bar
        attron(COLOR_PAIR(CP_HEADER));
        mvhline(maxY - 2, 0, ACS_HLINE, maxX);
        attroff(COLOR_PAIR(CP_HEADER));

        std::string status = " Selected: " + std::to_string(selectedCount) +
            "  [Up/Dn] Nav  [SPACE] Toggle  [/] Filter  [g] Grep  [ENTER] Diff  [a] All  [n] None  [q] Quit";
        mvprintw(maxY - 1, 0, "%-*s", maxX, status.c_str());
    }

    // ─── Drawing: Diff View ─────────────────────────────────────────────

    void drawDiffView() {
        int maxY, maxX;
        getmaxyx(stdscr, maxY, maxX);

        size_t nFiles = mSelectedNames.size();
        if (nFiles == 0) return;

        // Column widths
        int keyColWidth = 36;
        int valColWidth = 20;

        // Compute max key path width from visible rows
        for (auto idx : mVisibleRows) {
            int len = (int)mDiffRows[idx].keyPath.size();
            if (len + 2 > keyColWidth) keyColWidth = len + 2;
        }
        if (keyColWidth > maxX / 3) keyColWidth = maxX / 3;

        // Title
        attron(COLOR_PAIR(CP_HEADER) | A_BOLD);
        std::string title = " Diff: ";
        for (size_t i = 0; i < nFiles; ++i) {
            if (i > 0) title += " | ";
            title += mSelectedNames[i];
        }
        if (mDiffsOnly) title += "  [DIFFS ONLY]";
        else title += "  [ALL KEYS]";
        mvprintw(0, 0, "%-*s", maxX, title.c_str());
        attroff(COLOR_PAIR(CP_HEADER) | A_BOLD);

        // Column headers
        attron(COLOR_PAIR(CP_HEADER));
        mvhline(1, 0, ACS_HLINE, maxX);

        // Header row
        std::string hdr;
        hdr = padRight("Key Path", keyColWidth);
        // visible file columns with horizontal pan
        int availWidth = maxX - keyColWidth - 1;
        int visibleCols = std::max(1, availWidth / (valColWidth + 1));
        int maxPan = std::max(0, (int)nFiles - visibleCols);
        if (mColPan > maxPan) mColPan = maxPan;
        if (mColPan < 0) mColPan = 0;

        for (int c = 0; c < visibleCols && (mColPan + c) < (int)nFiles; ++c) {
            hdr += "|";
            std::string name = mSelectedNames[mColPan + c];
            // truncate name to fit
            if ((int)name.size() > valColWidth - 1)
                name = name.substr(0, valColWidth - 1);
            hdr += padRight(" " + name, valColWidth);
        }
        mvprintw(2, 0, "%-*s", maxX, hdr.c_str());
        mvhline(3, 0, ACS_HLINE, maxX);
        attroff(COLOR_PAIR(CP_HEADER));

        // Diff rows
        int rowStart = 4;
        int rowHeight = maxY - rowStart - 2;
        if (rowHeight < 1) rowHeight = 1;

        // Ensure cursor and scroll are in bounds
        if (mDiffCursor < 0) mDiffCursor = 0;
        if (mDiffCursor >= (int)mVisibleRows.size())
            mDiffCursor = std::max(0, (int)mVisibleRows.size() - 1);
        if (mDiffCursor < mDiffScroll) mDiffScroll = mDiffCursor;
        if (mDiffCursor >= mDiffScroll + rowHeight) mDiffScroll = mDiffCursor - rowHeight + 1;
        if (mDiffScroll < 0) mDiffScroll = 0;

        for (int i = 0; i < rowHeight; ++i) {
            int ridx = mDiffScroll + i;
            if (ridx >= (int)mVisibleRows.size()) break;

            auto& row = mDiffRows[mVisibleRows[ridx]];
            bool isCursorRow = (ridx == mDiffCursor);

            // Choose color
            int cp = CP_SAME;
            if (row.anyAbsent) cp = CP_ABSENT;
            else if (!row.allSame) cp = CP_DIFFER;

            if (isCursorRow) attron(A_REVERSE);

            // Key path column
            attron(COLOR_PAIR(CP_HEADER));
            std::string keyStr = padRight(row.keyPath, keyColWidth);
            mvprintw(rowStart + i, 0, "%s", keyStr.c_str());
            attroff(COLOR_PAIR(CP_HEADER));

            // Value columns
            attron(COLOR_PAIR(cp));
            for (int c = 0; c < visibleCols && (mColPan + c) < (int)nFiles; ++c) {
                int colX = keyColWidth + c * (valColWidth + 1);

                // separator
                attron(COLOR_PAIR(CP_HEADER));
                mvaddch(rowStart + i, colX, '|');
                attroff(COLOR_PAIR(CP_HEADER));

                attron(COLOR_PAIR(cp));
                std::string val = row.values[mColPan + c];
                if ((int)val.size() > valColWidth - 1)
                    val = val.substr(0, valColWidth - 2) + "~";
                mvprintw(rowStart + i, colX + 1, "%-*s", valColWidth - 1, val.c_str());
            }
            attroff(COLOR_PAIR(cp));
            if (isCursorRow) attroff(A_REVERSE);
        }

        // Status bar
        attron(COLOR_PAIR(CP_HEADER));
        mvhline(maxY - 2, 0, ACS_HLINE, maxX);
        attroff(COLOR_PAIR(CP_HEADER));

        std::string info = " Rows: " + std::to_string(mVisibleRows.size()) + "/" +
            std::to_string(mDiffRows.size());
        if ((int)nFiles > visibleCols)
            info += "  Cols: " + std::to_string(mColPan + 1) + "-" +
                std::to_string(std::min((int)nFiles, mColPan + visibleCols)) +
                "/" + std::to_string(nFiles);

        std::string help = "  [Up/Dn] Scroll  [Left/Right] Pan  [l] Expand  [d] Toggle diffs  [ESC] Back  [q] Quit";
        mvprintw(maxY - 1, 0, "%-*s", maxX, (info + help).c_str());
    }

    // ─── Input: File List ───────────────────────────────────────────────

    void handleFileListInput(int ch) {
        if (mFilterMode) {
            handleFilterInput(ch);
            return;
        }
        if (mGrepMode) {
            handleGrepInput(ch);
            return;
        }

        int maxY, maxX;
        getmaxyx(stdscr, maxY, maxX);
        int listHeight = maxY - 6;
        if (listHeight < 1) listHeight = 1;

        switch (ch) {
        case KEY_UP:
            if (mCursor > 0) mCursor--;
            break;
        case KEY_DOWN:
            if (mCursor < (int)mFiltered.size() - 1) mCursor++;
            break;
        case KEY_PPAGE:
            mCursor = std::max(0, mCursor - listHeight);
            break;
        case KEY_NPAGE:
            mCursor = std::min((int)mFiltered.size() - 1, mCursor + listHeight);
            break;
        case KEY_HOME:
            mCursor = 0;
            break;
        case KEY_END:
            mCursor = std::max(0, (int)mFiltered.size() - 1);
            break;
        case ' ': // toggle selection
            if (!mFiltered.empty()) {
                int fileIdx = mFiltered[mCursor];
                mAllFiles[fileIdx].selected = !mAllFiles[fileIdx].selected;
                if (mCursor < (int)mFiltered.size() - 1) mCursor++;
            }
            break;
        case '/': // enter filename filter mode
            mFilterMode = true;
            break;
        case 'g': case 'G': // enter content grep mode
            mGrepMode = true;
            break;
        case 'a': case 'A': // select all visible
            for (auto idx : mFiltered)
                mAllFiles[idx].selected = true;
            break;
        case 'n': case 'N': // deselect all
            for (auto& f : mAllFiles)
                f.selected = false;
            break;
        case '\n': case KEY_ENTER: { // open diff
            int count = 0;
            for (auto& f : mAllFiles) if (f.selected) count++;
            if (count >= 2) {
                loadSelectedFiles();
                computeDiff();
                mDiffScroll = 0;
                mDiffCursor = 0;
                mColPan = 0;
                mStage = DIFF_VIEW;
            }
            break;
        }
        }
    }

    void handleFilterInput(int ch) {
        if (ch == 27) { // ESC - clear filter and exit filter mode
            mFilterStr.clear();
            mFilterMode = false;
            applyFilter();
        } else if (ch == '\n' || ch == KEY_ENTER) {
            mFilterMode = false;
        } else if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
            if (!mFilterStr.empty()) {
                mFilterStr.pop_back();
                applyFilter();
            }
        } else if (ch >= 32 && ch < 127) {
            mFilterStr += (char)ch;
            applyFilter();
        }
    }

    void handleGrepInput(int ch) {
        if (ch == 27) { // ESC - clear grep and exit grep mode
            mGrepStr.clear();
            mGrepMode = false;
            applyFilter();
        } else if (ch == '\n' || ch == KEY_ENTER) {
            mGrepMode = false;
        } else if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
            if (!mGrepStr.empty()) {
                mGrepStr.pop_back();
                applyFilter();
            }
        } else if (ch >= 32 && ch < 127) {
            mGrepStr += (char)ch;
            applyFilter();
        }
    }

    // ─── Input: Diff View ───────────────────────────────────────────────

    void handleDiffViewInput(int ch) {
        int maxY, maxX;
        getmaxyx(stdscr, maxY, maxX);
        int rowHeight = maxY - 6;
        if (rowHeight < 1) rowHeight = 1;
        int lastRow = std::max(0, (int)mVisibleRows.size() - 1);

        switch (ch) {
        case KEY_UP:
            if (mDiffCursor > 0) mDiffCursor--;
            break;
        case KEY_DOWN:
            if (mDiffCursor < lastRow) mDiffCursor++;
            break;
        case KEY_PPAGE:
            mDiffCursor = std::max(0, mDiffCursor - rowHeight);
            break;
        case KEY_NPAGE:
            mDiffCursor = std::min(lastRow, mDiffCursor + rowHeight);
            break;
        case KEY_HOME:
            mDiffCursor = 0;
            break;
        case KEY_END:
            mDiffCursor = lastRow;
            break;
        case KEY_LEFT:
            if (mColPan > 0) mColPan--;
            break;
        case KEY_RIGHT:
            mColPan++;
            break;
        case 'l': case 'L':
            showEntryPopup();
            break;
        case 'd': case 'D':
            mDiffsOnly = !mDiffsOnly;
            updateVisibleRows();
            break;
        case 27: // ESC - back to file list
            mStage = FILE_LIST;
            break;
        }
    }

    // ─── Popup: Full entry view ─────────────────────────────────────────

    void showEntryPopup() {
        if (mVisibleRows.empty()) return;
        if (mDiffCursor < 0 || mDiffCursor >= (int)mVisibleRows.size()) return;

        auto& row = mDiffRows[mVisibleRows[mDiffCursor]];
        size_t nFiles = mSelectedNames.size();

        // Compute popup dimensions
        int maxY, maxX;
        getmaxyx(stdscr, maxY, maxX);

        // Build lines for display
        std::vector<std::pair<int, std::string>> lines; // color pair, text
        lines.push_back({CP_POPUP_HDR, " Key: " + row.keyPath});
        lines.push_back({CP_POPUP, ""});

        int maxWidth = (int)row.keyPath.size() + 7;

        for (size_t i = 0; i < nFiles; ++i) {
            int cp = CP_SAME;
            if (row.values[i] == "[absent]") cp = CP_ABSENT;
            else if (!row.allSame) cp = CP_DIFFER;

            std::string line = " " + mSelectedNames[i] + ":";
            std::string val  = "   " + row.values[i];

            if ((int)line.size() > maxWidth) maxWidth = (int)line.size();
            if ((int)val.size() > maxWidth) maxWidth = (int)val.size();

            lines.push_back({CP_HEADER, line});
            lines.push_back({cp, val});
        }
        lines.push_back({CP_POPUP, ""});
        lines.push_back({CP_POPUP, " Press any key to close"});

        // Size the popup window
        int popH = std::min((int)lines.size() + 2, maxY - 4);
        int popW = std::min(maxWidth + 4, maxX - 4);
        if (popW < 30) popW = 30;
        int popY = (maxY - popH) / 2;
        int popX = (maxX - popW) / 2;

        WINDOW* popup = newwin(popH, popW, popY, popX);
        wbkgd(popup, COLOR_PAIR(CP_POPUP));
        box(popup, 0, 0);

        // Title bar
        wattron(popup, COLOR_PAIR(CP_POPUP_HDR) | A_BOLD);
        mvwprintw(popup, 0, 2, " Entry Detail ");
        wattroff(popup, COLOR_PAIR(CP_POPUP_HDR) | A_BOLD);

        // Content (scrollable if needed)
        int contentH = popH - 2;
        int contentW = popW - 2;
        for (int i = 0; i < contentH && i < (int)lines.size(); ++i) {
            wattron(popup, COLOR_PAIR(lines[i].first));
            std::string display = lines[i].second;
            if ((int)display.size() > contentW)
                display = display.substr(0, contentW - 1) + "~";
            mvwprintw(popup, i + 1, 1, "%-*s", contentW, display.c_str());
            wattroff(popup, COLOR_PAIR(lines[i].first));
        }

        wrefresh(popup);
        wgetch(popup);
        delwin(popup);
    }

    // ─── Utilities ──────────────────────────────────────────────────────

    static std::string toLower(const std::string& s) {
        std::string r = s;
        for (auto& c : r) c = std::tolower((unsigned char)c);
        return r;
    }

    static std::string padRight(const std::string& s, int width) {
        if ((int)s.size() >= width) return s.substr(0, width);
        return s + std::string(width - s.size(), ' ');
    }

    // ─── State ──────────────────────────────────────────────────────────

    std::string mDir;
    Stage mStage = FILE_LIST;

    // File list state
    std::vector<YamlFileEntry> mAllFiles;
    std::vector<int> mFiltered; // indices into mAllFiles
    int mCursor = 0;
    int mScrollOffset = 0;
    bool mFilterMode = false;
    std::string mFilterStr;

    // Content grep state
    bool mGrepMode = false;
    std::string mGrepStr;
    std::vector<std::string> mContentCache; // lowercased raw file contents
    bool mContentCached = false;

    // Diff state
    std::vector<std::string> mSelectedNames;
    std::vector<std::map<std::string, std::string>> mFlatMaps;
    std::vector<DiffRow> mDiffRows;
    std::vector<int> mVisibleRows; // indices into mDiffRows
    int mDiffScroll = 0;
    int mDiffCursor = 0; // index into mVisibleRows
    int mColPan = 0;
    bool mDiffsOnly = true;
};

// ─── main ───────────────────────────────────────────────────────────────────

static void printUsage(const char* prog) {
    fprintf(stderr, "Usage: %s [--dir <path>]\n", prog);
    fprintf(stderr, "  Default directory: " PROJECT_ROOT "/data/env\n");
}

int main(int argc, char* argv[]) {
    std::string dir = std::string(PROJECT_ROOT) + "/data/env";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--dir" || arg == "-d") && i + 1 < argc) {
            dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            printUsage(argv[0]);
            return 1;
        }
    }

    if (!fs::is_directory(dir)) {
        fprintf(stderr, "Error: '%s' is not a directory\n", dir.c_str());
        return 1;
    }

    YamlDiffUI ui(dir);
    ui.run();
    return 0;
}
