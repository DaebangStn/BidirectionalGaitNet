#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace PMuscle {

struct FTPCredentials {
    std::string ip;
    std::string username;
    std::string password;
    int port;
};

class URIResolver {
public:
    // Singleton access
    static URIResolver& getInstance();

    // Initialize the resolver with default schemes
    void initialize();

    // Resolve a URI to an absolute path
    // @data/file.xml -> /path/to/project/data/file.xml
    // @ftp:hostname/path -> downloads and returns temp file path
    std::string resolve(const std::string& uri) const;

    // Check if a string is a URI (starts with @)
    bool isURI(const std::string& path) const;

    // Register a new scheme with its root path
    void registerScheme(const std::string& scheme, const std::string& rootPath);

private:
    URIResolver() = default;
    ~URIResolver() = default;
    URIResolver(const URIResolver&) = delete;
    URIResolver& operator=(const URIResolver&) = delete;

    // Get the data root path (from compile-time or fallback)
    std::string getDataRootPath() const;

    // FTP-related methods
    std::string resolveFTP(const std::string& host, const std::string& path) const;
    std::string resolveWildcardPath(const std::string& host, const std::string& path, const FTPCredentials& creds) const;
    std::vector<std::string> listFTPDirectory(const std::string& host, const std::string& path, const FTPCredentials& creds, bool directoriesOnly = true) const;
    FTPCredentials loadFTPCredentials(const std::string& host) const;
    bool checkFTPFileExists(const std::string& url, const FTPCredentials& creds) const;
    std::string downloadFTPFile(const std::string& url, const FTPCredentials& creds, const std::string& destPath) const;
    std::string getTempDir() const;
    std::string generateTempFilePath(const std::string& host, const std::string& remotePath) const;

    bool mInitialized = false;
    std::unordered_map<std::string, std::string> mSchemeRoots;
    mutable std::unordered_map<std::string, FTPCredentials> mFTPCredentialsCache;
};

} // namespace PMuscle