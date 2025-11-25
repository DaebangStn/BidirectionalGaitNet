#include "MotionProcessorFactory.h"
#include <algorithm>
#include <cctype>

std::string MotionProcessorFactory::getTypeForExtension(const std::string& extension)
{
    // Convert to lowercase for comparison
    std::string ext = extension;
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // HDF5 extensions
    if (ext == ".h5" || ext == ".hdf5")
        return "hdf";

    // C3D extension
    if (ext == ".c3d")
        return "c3d";

    return "";  // Unknown extension
}

bool MotionProcessorFactory::isExtensionSupported(const std::string& extension)
{
    return !getTypeForExtension(extension).empty();
}

std::vector<std::string> MotionProcessorFactory::getSupportedExtensions()
{
    return {".h5", ".hdf5", ".c3d"};
}

std::string MotionProcessorFactory::extractExtension(const std::string& path)
{
    size_t dotPos = path.rfind('.');
    if (dotPos == std::string::npos)
        return "";

    std::string ext = path.substr(dotPos);

    // Convert to lowercase
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    return ext;
}
