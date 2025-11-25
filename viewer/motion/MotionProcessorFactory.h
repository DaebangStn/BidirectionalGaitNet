#pragma once

#include <memory>
#include <string>
#include <vector>

class IMotionProcessor;

/**
 * @brief Factory for creating motion processors
 *
 * Provides centralized creation of motion processors based on
 * file extension or explicit type specification.
 */
class MotionProcessorFactory {
public:
    /**
     * @brief Get processor type for file extension
     * @param extension File extension (e.g., ".h5", ".c3d")
     * @return Processor type ("hdf", "c3d") or empty string if unknown
     */
    static std::string getTypeForExtension(const std::string& extension);

    /**
     * @brief Check if extension is supported
     * @param extension File extension to check
     * @return true if extension maps to a known processor type
     */
    static bool isExtensionSupported(const std::string& extension);

    /**
     * @brief Get all supported file extensions
     * @return Vector of supported extensions
     */
    static std::vector<std::string> getSupportedExtensions();

    /**
     * @brief Extract extension from file path
     * @param path File path
     * @return Extension (lowercase, including dot)
     */
    static std::string extractExtension(const std::string& path);
};
