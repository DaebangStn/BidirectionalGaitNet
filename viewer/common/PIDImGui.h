#pragma once

#include "PIDFileFilter.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rm {
class ResourceManager;
}

namespace PIDNav {

/**
 * POD structure holding PID selection state.
 * Maintains the list of PIDs with their metadata and current selection.
 */
struct PIDSelectionState {
    std::vector<std::string> pidList;
    std::vector<std::string> pidNames;
    std::vector<std::string> pidGMFCS;
    std::vector<std::vector<std::string>> pidVisits;  // Available visits per PID
    int selectedPID = -1;
    int selectedVisit = 0;  // Index into pidVisits[selectedPID]

    // Legacy compatibility
    bool preOp = true;  // Derived from selectedVisit == 0

    /**
     * Returns the currently selected PID string, or empty if none selected.
     */
    std::string getSelectedPID() const;

    /**
     * Returns the selected visit directory name (e.g., "pre", "op1", "op2").
     */
    std::string getVisitDir() const;

    /**
     * Returns "pre" or "post" based on current visit (legacy compatibility).
     * @deprecated Use getVisitDir() instead.
     */
    std::string getPrePostDir() const;
};

/**
 * Callback invoked when user selects a file.
 * Parameters: (resolvedPath, filename)
 */
using FileSelectionCallback = std::function<void(
    const std::string& resolvedPath,
    const std::string& filename
)>;

/**
 * Callback invoked when a file is deleted from the list.
 * Parameter: filename (the deleted file name)
 */
using FileDeleteCallback = std::function<void(const std::string& filename)>;

/**
 * Callback invoked when PID selection changes.
 * Parameter: pid (the selected PID string)
 */
using PIDChangeCallback = std::function<void(const std::string& pid)>;
using VisitChangeCallback = std::function<void(const std::string& pid, const std::string& visit)>;

/**
 * Configuration for registering a file type section.
 */
struct FileTypeConfig {
    std::string label;                    // Display label (e.g., "Skeleton", "Muscle")
    std::unique_ptr<FileFilter> filter;   // File filter strategy
    FileSelectionCallback onSelect;       // Called on file selection
    FileDeleteCallback onDelete;          // Called when Del is clicked (optional)
};

/**
 * PID Navigator component for browsing clinical data.
 *
 * Manages PID list, file browsing with pluggable file type filters,
 * and renders ImGui UI for selection. Uses dependency injection for
 * ResourceManager and strategy pattern for file filtering.
 *
 * Usage:
 *   auto nav = std::make_unique<PIDNavigator>(
 *       resourceManager,
 *       std::make_unique<HDFFileFilter>()
 *   );
 *   nav->setFileSelectionCallback([](auto path, auto name) {
 *       loadFile(path);
 *   });
 *   nav->scanPIDs();
 *
 *   // In UI rendering:
 *   nav->renderUI("Clinical Data");
 */
class PIDNavigator {
public:
    /**
     * Constructs PID navigator with dependency injection.
     * Use registerFileType() to add file type sections.
     *
     * @param rm Non-owning pointer to ResourceManager singleton
     */
    explicit PIDNavigator(rm::ResourceManager* rm);

    /**
     * Legacy constructor for single file type (backward compatibility).
     *
     * @param rm Non-owning pointer to ResourceManager singleton
     * @param filter File filter strategy (ownership transferred)
     * @deprecated Use the single-argument constructor with registerFileType() instead.
     */
    PIDNavigator(rm::ResourceManager* rm,
                 std::unique_ptr<FileFilter> filter);

    ~PIDNavigator();

    // Non-copyable, movable
    PIDNavigator(const PIDNavigator&) = delete;
    PIDNavigator& operator=(const PIDNavigator&) = delete;
    PIDNavigator(PIDNavigator&&) noexcept;
    PIDNavigator& operator=(PIDNavigator&&) noexcept;

    /**
     * Registers a file type section to be displayed in the navigator.
     * Multiple file types can be registered (e.g., Skeleton, Muscle, Motion).
     *
     * @param config Configuration including label, filter, and callbacks
     */
    void registerFileType(FileTypeConfig config);

    /**
     * Scans the resource manager for available PIDs and their metadata.
     * Populates the PID list, names, and GMFCS levels.
     */
    void scanPIDs();

    /**
     * Scans for files of all registered types in the specified PID/visit directory.
     *
     * @param pid The PID to scan (e.g., "12964246")
     * @param visit The visit directory (e.g., "pre", "op1", "op2")
     */
    void scanFiles(const std::string& pid, const std::string& visit);

    /**
     * Returns the current PID selection state (read-only).
     */
    const PIDSelectionState& getState() const;

    /**
     * Returns the current file list for the primary (first registered) file type (read-only).
     * @deprecated Use getFiles(label) for multi-file-type support.
     */
    const std::vector<std::string>& getFiles() const;

    /**
     * Returns the file list for a specific file type section.
     *
     * @param label The label of the file type section (e.g., "Skeleton", "Muscle")
     * @return The file list, or empty vector if label not found
     */
    const std::vector<std::string>& getFiles(const std::string& label) const;

    /**
     * Sets callback to invoke when user selects a file (legacy, for primary file type).
     * @deprecated Use registerFileType() with onSelect callback instead.
     */
    void setFileSelectionCallback(FileSelectionCallback callback);

    /**
     * Sets callback to invoke when PID selection changes.
     */
    void setPIDChangeCallback(PIDChangeCallback callback);

    /**
     * Sets callback to invoke when visit selection changes.
     */
    void setVisitChangeCallback(VisitChangeCallback callback);

    /**
     * Programmatically navigate to a specific PID and visit.
     * Useful for initializing from config files.
     *
     * @param pid The PID to select (e.g., "29792292")
     * @param visit The visit to select (e.g., "pre", "op1")
     * @return true if navigation was successful, false if PID not found
     */
    bool navigateTo(const std::string& pid, const std::string& visit = "pre");

    /**
     * Renders PID navigator UI.
     *
     * @param title Section title for collapsing header. Pass nullptr to render without header.
     * @param pidListHeight Height of PID list box in pixels
     * @param fileSectionHeight Height per file section in pixels (0 to hide file sections)
     * @param defaultOpen Whether the section is open by default (ignored if title is nullptr)
     */
    void renderUI(const char* title = "Clinical Data",
                  float pidListHeight = 150.0f,
                  float fileSectionHeight = 100.0f,
                  bool defaultOpen = false);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace PIDNav
