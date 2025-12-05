#include "UriResolver.h"
#include "Log.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <curl/curl.h>
#include <yaml-cpp/yaml.h>

namespace PMuscle {

// Static data root path - computed once at startup
static std::string computeDataRoot() {
    // Get the data root path from compile-time define or fallback
    #ifdef DATA_ROOT_PATH
        return DATA_ROOT_PATH;
    #else
        // Fallback: try to find data directory relative to executable or current directory
        std::filesystem::path cwd = std::filesystem::current_path();
        std::filesystem::path dataPath = cwd / "data";
        if (std::filesystem::exists(dataPath)) {
            return dataPath.string();
        }
        // Last resort: return current directory
        return cwd.string();
    #endif
}

// Global constant - initialized before main() runs, fully thread-safe
static const std::string g_dataRoot = computeDataRoot();

URIResolver::URIResolver() : mDataRoot(g_dataRoot) {
    // mDataRoot is a copy of the global constant, fully initialized before any threads start
    mSchemeRoots["data"] = mDataRoot;
    LOG_VERBOSE("[URIResolver] Initialized with data root: " << mDataRoot);
}

URIResolver& URIResolver::getInstance() {
    static URIResolver instance;
    return instance;
}

void URIResolver::initialize() {
    // No-op: initialization now happens in constructor
    // Kept for backward compatibility with existing code that calls initialize()
}

std::string URIResolver::resolve(const std::string& uri) const {
    if (!isURI(uri)) {
        // Special case: check for */filename pattern and resolve to data/filename
        if (!uri.empty() && uri[0] == '*' && uri.length() > 1 && uri[1] == '/') {
            std::string filename = uri.substr(2); // Remove */
            std::filesystem::path fullPath = std::filesystem::path(mDataRoot) / filename;
            return fullPath.string();
        }

        // Backwards compatibility: check for ../data/ pattern and resolve to data/
        if (uri.find("../data/") == 0) {
            std::string filename = uri.substr(8); // Remove "../data/"
            std::filesystem::path fullPath = std::filesystem::path(mDataRoot) / filename;
            LOG_VERBOSE("URIResolver: Backwards compatibility - resolving " << uri << " to " << fullPath.string());
            return fullPath.string();
        }

        return uri; // Not a URI, return as-is
    }

    std::string scheme, relativePath;

    // Handle @scheme:path or @scheme/path formats
    if (uri[0] == '@') {
        // Check for @ftp:host/path format specifically
        if (uri.substr(1, 4) == "ftp:") {
            // @ftp:host/path format
            std::string hostPath = uri.substr(5); // Remove "@ftp:"
            size_t hostEnd = hostPath.find('/');
            if (hostEnd == std::string::npos) {
                std::printf("Error: Invalid FTP URI format (missing path): %s\n", uri.c_str());
                std::exit(1);
            }
            std::string host = hostPath.substr(0, hostEnd);
            std::string path = hostPath.substr(hostEnd); // Include leading /
            return resolveFTP(host, path);
        }

        // @scheme/path format for other schemes
        size_t schemeEnd = uri.find('/', 1);
        if (schemeEnd == std::string::npos) {
            std::printf("Warning: Invalid URI format: %s\n", uri.c_str());
            return uri;
        }
        scheme = uri.substr(1, schemeEnd - 1); // Remove @ and get scheme
        relativePath = uri.substr(schemeEnd + 1); // Get path after scheme/
    } else {
        // scheme:path format
        size_t colonPos = uri.find(':');
        if (colonPos == std::string::npos) {
            std::printf("Warning: Invalid URI format: %s\n", uri.c_str());
            return uri;
        }
        scheme = uri.substr(0, colonPos);
        relativePath = uri.substr(colonPos + 1);
    }

    // Only "data" scheme is supported, use mDataRoot directly
    if (scheme != "data") {
        std::printf("Warning: Unknown URI scheme: %s\n", scheme.c_str());
        return uri;
    }

    std::filesystem::path fullPath = std::filesystem::path(mDataRoot) / relativePath;
    return fullPath.string();
}

bool URIResolver::isURI(const std::string& path) const {
    if (path.empty()) return false;
    
    // Check for @scheme/path format
    if (path[0] == '@') return true;
    
    // Check for scheme:path format (must have colon but not be an absolute path)
    size_t colonPos = path.find(':');
    if (colonPos != std::string::npos && colonPos > 0) {
        // Make sure it's not a Windows drive letter (C:) or absolute path
        if (colonPos == 1 && path.length() > 2 && (path[2] == '\\' || path[2] == '/')) {
            return false; // Windows drive letter
        }
        // Make sure scheme part doesn't contain path separators
        std::string scheme = path.substr(0, colonPos);
        return scheme.find('/') == std::string::npos && scheme.find('\\') == std::string::npos;
    }
    
    return false;
}

void URIResolver::registerScheme(const std::string& scheme, const std::string& rootPath) {
    mSchemeRoots[scheme] = rootPath;
}

std::string URIResolver::getDataRootPath() const {
#ifdef PROJECT_ROOT
    std::filesystem::path projectRoot = PROJECT_ROOT;
    std::filesystem::path dataPath = projectRoot / "data";
    return dataPath.string();
#elif defined(DATA_ROOT_PATH)
    return DATA_ROOT_PATH;
#else
    // Fallback to current working directory
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path dataPath = currentPath / "data";
    return dataPath.string();
#endif
}

std::string URIResolver::getTempDir() const {
#ifdef PROJECT_ROOT
    std::filesystem::path projectRoot = PROJECT_ROOT;
    std::filesystem::path tempPath = projectRoot / ".temp";
#else
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path tempPath = currentPath / ".temp";
#endif

    // Create .temp directory if it doesn't exist
    if (!std::filesystem::exists(tempPath)) {
        std::filesystem::create_directories(tempPath);
    }

    return tempPath.string();
}

std::string URIResolver::generateTempFilePath(const std::string& host, const std::string& remotePath) const {
    std::string tempDir = getTempDir();

    // Convert remote path to safe filename: /path/to/file.xml -> path_to_file.xml
    std::string safeFilename = remotePath;
    std::replace(safeFilename.begin(), safeFilename.end(), '/', '_');
    if (!safeFilename.empty() && safeFilename[0] == '_') {
        safeFilename = safeFilename.substr(1);
    }

    // Prepend host to avoid conflicts: gait_path_to_file.xml
    std::string fullFilename = "ftp_" + host + "_" + safeFilename;

    std::filesystem::path tempFilePath = std::filesystem::path(tempDir) / fullFilename;
    return tempFilePath.string();
}

FTPCredentials URIResolver::loadFTPCredentials(const std::string& host) const {
    // Check cache first
    auto it = mFTPCredentialsCache.find(host);
    if (it != mFTPCredentialsCache.end()) {
        return it->second;
    }

    // Load from secret.yaml
    std::string secretPath;
#ifdef PROJECT_ROOT
    std::filesystem::path projectRoot = PROJECT_ROOT;
    secretPath = (projectRoot / "data" / "secret.yaml").string();
#else
    std::filesystem::path currentPath = std::filesystem::current_path();
    secretPath = (currentPath / "data" / "secret.yaml").string();
#endif

    try {
        YAML::Node config = YAML::LoadFile(secretPath);

        if (config[host]) {
            FTPCredentials creds;
            creds.ip = config[host]["ip"].as<std::string>();
            creds.username = config[host]["username"].as<std::string>();
            creds.password = config[host]["password"].as<std::string>();
            creds.port = config[host]["port"].as<int>();

            // Cache for future use
            mFTPCredentialsCache[host] = creds;

            return creds;
        } else {
            std::printf("Warning: FTP host '%s' not found in secret.yaml\n", host.c_str());
        }
    } catch (const YAML::Exception& e) {
        std::printf("Error loading FTP credentials: %s\n", e.what());
    }

    // Return empty credentials on error
    return FTPCredentials();
}

// CURL write callback for downloading files
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;
    std::ofstream* file = static_cast<std::ofstream*>(userp);
    file->write(static_cast<char*>(contents), totalSize);
    return totalSize;
}

// CURL write callback for directory listing
static size_t ListCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;
    std::string* str = static_cast<std::string*>(userp);
    str->append(static_cast<char*>(contents), totalSize);
    return totalSize;
}

// CURL write callback for checking file existence (discards data)
static size_t DiscardCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    return size * nmemb;  // Just return size without storing
}

bool URIResolver::checkFTPFileExists(const std::string& url, const FTPCredentials& creds) const {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return false;
    }

    bool exists = false;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);  // HEAD request
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, DiscardCallback);

    // Set credentials
    std::string userpass = creds.username + ":" + creds.password;
    curl_easy_setopt(curl, CURLOPT_USERPWD, userpass.c_str());

    CURLcode res = curl_easy_perform(curl);

    if (res == CURLE_OK) {
        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        exists = (response_code == 213 || response_code == 0);  // FTP file exists
    }

    curl_easy_cleanup(curl);
    return exists;
}

std::string URIResolver::downloadFTPFile(const std::string& url, const FTPCredentials& creds, const std::string& destPath) const {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::printf("Error: Failed to initialize CURL\n");
        return "";
    }

    std::ofstream outFile(destPath, std::ios::binary);
    if (!outFile.is_open()) {
        std::printf("Error: Cannot open file for writing: %s\n", destPath.c_str());
        curl_easy_cleanup(curl);
        return "";
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &outFile);

    // Set credentials
    std::string userpass = creds.username + ":" + creds.password;
    curl_easy_setopt(curl, CURLOPT_USERPWD, userpass.c_str());

    // Perform download
    CURLcode res = curl_easy_perform(curl);
    outFile.close();

    if (res != CURLE_OK) {
        std::printf("Error: FTP download failed: %s\n", curl_easy_strerror(res));
        std::filesystem::remove(destPath);  // Clean up partial file
        curl_easy_cleanup(curl);
        return "";
    }

    curl_easy_cleanup(curl);
    std::printf("FTP: Downloaded %s to %s\n", url.c_str(), destPath.c_str());
    return destPath;
}

std::vector<std::string> URIResolver::listFTPDirectory(const std::string& host, const std::string& path, const FTPCredentials& creds, bool directoriesOnly) const {
    std::vector<std::string> directories;
    std::string ftpUrl = "ftp://" + creds.ip + ":" + std::to_string(creds.port) + path;

    // Add trailing slash for directory listing (required by CURL)
    if (!ftpUrl.empty() && ftpUrl.back() != '/') {
        ftpUrl += "/";
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::printf("Error: Failed to initialize CURL for directory listing\n");
        std::exit(1);
    }

    std::string listing;
    curl_easy_setopt(curl, CURLOPT_URL, ftpUrl.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ListCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &listing);

    // Set credentials
    std::string userpass = creds.username + ":" + creds.password;
    curl_easy_setopt(curl, CURLOPT_USERPWD, userpass.c_str());

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::printf("Error: FTP directory listing failed for path '%s': %s\n",
                   path.c_str(), curl_easy_strerror(res));
        std::printf("       Full URL: %s\n", ftpUrl.c_str());
        std::exit(1);
    }

    // Parse directory listing - extract names
    std::istringstream iss(listing);
    std::string line;
    while (std::getline(iss, line)) {
        // Skip empty lines
        if (line.empty()) continue;

        // Parse entries (directories start with 'd', files start with '-')
        bool isDirectory = (line[0] == 'd');
        bool isFile = (line[0] == '-');

        if (isDirectory || (!directoriesOnly && isFile)) {
            // Extract last token (file/directory name)
            size_t lastSpace = line.find_last_of(" \t");
            if (lastSpace != std::string::npos) {
                std::string name = line.substr(lastSpace + 1);
                // Remove trailing /
                if (!name.empty() && name.back() == '/') {
                    name.pop_back();
                }
                if (!name.empty() && name != "." && name != "..") {
                    directories.push_back(name);
                }
            }
        }
    }

    // Sort directories in descending order (latest first)
    std::sort(directories.begin(), directories.end(), std::greater<std::string>());

    return directories;
}

std::string URIResolver::resolveWildcardPath(const std::string& host, const std::string& path, const FTPCredentials& creds) const {
    // Split path by / and process each segment
    std::vector<std::string> segments;
    std::istringstream iss(path);
    std::string segment;

    while (std::getline(iss, segment, '/')) {
        if (!segment.empty()) {
            segments.push_back(segment);
        }
    }

    std::string resolvedPath = "";

    for (size_t segIdx = 0; segIdx < segments.size(); ++segIdx) {
        const auto& seg = segments[segIdx];
        bool isLastSegment = (segIdx == segments.size() - 1);
        // Check if segment contains #n wildcard
        if (seg.find("#") != std::string::npos) {
            // Extract number after #
            size_t hashPos = seg.find("#");
            size_t endPos = hashPos + 1;
            while (endPos < seg.length() && std::isdigit(seg[endPos])) {
                endPos++;
            }

            if (endPos == hashPos + 1) {
                std::printf("Error: Invalid #n format in segment: %s\n", seg.c_str());
                std::exit(1);
            }

            int depth = std::stoi(seg.substr(hashPos + 1, endPos - hashPos - 1));

            // Descend n levels from current path (always directories only)
            std::string currentPath = resolvedPath;
            for (int i = 0; i < depth; ++i) {
                std::vector<std::string> dirs = listFTPDirectory(host, currentPath, creds, true);
                if (dirs.empty()) {
                    std::printf("Error: No directories at '%s' (depth %d/%d)\n",
                               currentPath.c_str(), i+1, depth);
                    std::exit(1);
                }
                resolvedPath = resolvedPath + "/" + dirs[0];
                currentPath = resolvedPath;
                std::printf("FTP: #%d → %s\n", i+1, dirs[0].c_str());
            }
        }
        // Check if segment contains * wildcard
        else if (seg.find("*") != std::string::npos) {
            // Split segment by *
            size_t starPos = seg.find("*");
            std::string prefix = seg.substr(0, starPos);
            std::string suffix = seg.substr(starPos + 1);

            // List current directory and find match
            // If this is the last segment, include files; otherwise directories only
            std::printf("FTP: Listing for pattern '%s*%s' at: %s (last=%d)\n", prefix.c_str(), suffix.c_str(), resolvedPath.c_str(), isLastSegment);
            std::vector<std::string> dirs = listFTPDirectory(host, resolvedPath, creds, !isLastSegment);
            std::printf("FTP: Found %zu items\n", dirs.size());

            std::string matched;
            for (const auto& dir : dirs) {
                std::printf("FTP: Checking '%s' against prefix '%s'\n", dir.c_str(), prefix.c_str());
                if (dir.find(prefix) == 0) {
                    if (suffix.empty() || dir.find(suffix, prefix.length()) != std::string::npos) {
                        matched = dir;
                        std::printf("FTP: MATCHED!\n");
                        break;
                    }
                }
            }

            if (matched.empty()) {
                std::printf("Error: No match for '%s*%s' at '%s'\n",
                           prefix.c_str(), suffix.c_str(), resolvedPath.c_str());
                std::printf("       Available (%zu items): ", dirs.size());
                for (size_t i = 0; i < std::min(dirs.size(), size_t(5)); ++i) {
                    std::printf("'%s'%s", dirs[i].c_str(), (i < dirs.size() - 1) ? ", " : "");
                }
                std::printf("\n");
                std::exit(1);
            }

            resolvedPath = resolvedPath + "/" + matched;
            std::printf("FTP: %s*%s → %s\n", prefix.c_str(), suffix.c_str(), matched.c_str());
        }
        else {
            // Regular segment, just append
            resolvedPath = resolvedPath + "/" + seg;
        }
    }

    return resolvedPath;
}

std::string URIResolver::resolveFTP(const std::string& host, const std::string& path) const {
    // Load credentials for the host
    FTPCredentials creds = loadFTPCredentials(host);
    if (creds.ip.empty()) {
        std::printf("Error: No credentials found for FTP host: %s\n", host.c_str());
        std::exit(1);
    }

    // Resolve wildcards if present (# or *)
    std::string resolvedPath = path;
    if (path.find("#") != std::string::npos || path.find("*") != std::string::npos) {
        resolvedPath = resolveWildcardPath(host, path, creds);
        std::printf("FTP: Wildcard path %s resolved to %s\n", path.c_str(), resolvedPath.c_str());
    }

    // Construct FTP URL
    std::string ftpUrl = "ftp://" + creds.ip + ":" + std::to_string(creds.port) + resolvedPath;

    // Generate temp file path
    std::string tempFilePath = generateTempFilePath(host, resolvedPath);

    // Check if file already exists in temp (cached)
    if (std::filesystem::exists(tempFilePath)) {
        std::printf("FTP: Using cached file: %s\n", tempFilePath.c_str());
        return tempFilePath;
    }

    // Download the file (will fail with proper error if file doesn't exist)
    std::string downloadedPath = downloadFTPFile(ftpUrl, creds, tempFilePath);
    if (downloadedPath.empty()) {
        std::printf("Error: Failed to download FTP file: %s\n", ftpUrl.c_str());
        std::exit(1);
    }

    return downloadedPath;
}

} // namespace PMuscle