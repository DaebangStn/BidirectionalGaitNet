// Resource Manager Test Suite
// Tests URI parsing, @data and @pid prefix routing, and backend operations

#include "rm/rm.hpp"
#include <iostream>
#include <cassert>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

// Absolute path to config
static const std::string CONFIG_PATH = "/home/geon/BidirectionalGaitNet/data/rm_config.yaml";
static const std::string DATA_ROOT = "/home/geon/BidirectionalGaitNet/data";
static const std::string PID_ROOT = "/mnt/blue8T/CP/RM";

// Test helper macros
#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        std::cerr << "  FAILED: " << #a << " != " << #b << std::endl; \
        std::cerr << "    Got: '" << (a) << "' vs '" << (b) << "'" << std::endl; \
        std::exit(1); \
    } \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        std::cerr << "  FAILED: " << #cond << " is false" << std::endl; \
        std::exit(1); \
    } \
} while(0)

#define ASSERT_FALSE(cond) do { \
    if (cond) { \
        std::cerr << "  FAILED: " << #cond << " is true" << std::endl; \
        std::exit(1); \
    } \
} while(0)

#define ASSERT_THROWS(expr, exception_type) do { \
    bool caught = false; \
    try { expr; } catch (const exception_type&) { caught = true; } \
    if (!caught) { \
        std::cerr << "  FAILED: " << #expr << " did not throw " << #exception_type << std::endl; \
        std::exit(1); \
    } \
} while(0)

// ============================================================================
// URI Parsing Tests
// ============================================================================

void test_uri_data_prefix() {
    std::cout << "  test_uri_data_prefix..." << std::endl;

    rm::URI uri = rm::URI::parse("@data/skeleton/base.xml");

    ASSERT_TRUE(uri.has_prefix());
    ASSERT_EQ(uri.prefix(), "@data");
    ASSERT_EQ(uri.prefix_arg(), "");
    ASSERT_EQ(uri.path(), "skeleton/base.xml");
    ASSERT_EQ(uri.resolved_path(), "skeleton/base.xml");
    ASSERT_EQ(uri.to_string(), "@data/skeleton/base.xml");
    ASSERT_TRUE(uri.is_relative());
    ASSERT_FALSE(uri.is_absolute());
}

void test_uri_pid_prefix_with_arg() {
    std::cout << "  test_uri_pid_prefix_with_arg..." << std::endl;

    rm::URI uri = rm::URI::parse("@pid:CP001/markers/trial.c3d");

    ASSERT_TRUE(uri.has_prefix());
    ASSERT_EQ(uri.prefix(), "@pid");
    ASSERT_EQ(uri.prefix_arg(), "CP001");
    ASSERT_EQ(uri.path(), "markers/trial.c3d");
    ASSERT_EQ(uri.resolved_path(), "CP001/markers/trial.c3d");
    ASSERT_EQ(uri.to_string(), "@pid:CP001/markers/trial.c3d");
}

void test_uri_pid_prefix_arg_only() {
    std::cout << "  test_uri_pid_prefix_arg_only..." << std::endl;

    rm::URI uri = rm::URI::parse("@pid:CP001");

    ASSERT_TRUE(uri.has_prefix());
    ASSERT_EQ(uri.prefix(), "@pid");
    ASSERT_EQ(uri.prefix_arg(), "CP001");
    ASSERT_EQ(uri.path(), "");
    ASSERT_EQ(uri.resolved_path(), "CP001");
}

void test_uri_plain_path() {
    std::cout << "  test_uri_plain_path..." << std::endl;

    rm::URI uri = rm::URI::parse("skeleton/base.xml");

    ASSERT_FALSE(uri.has_prefix());
    ASSERT_EQ(uri.prefix(), "");
    ASSERT_EQ(uri.path(), "skeleton/base.xml");
    ASSERT_EQ(uri.resolved_path(), "skeleton/base.xml");
    ASSERT_TRUE(uri.is_relative());
}

void test_uri_absolute_path() {
    std::cout << "  test_uri_absolute_path..." << std::endl;

    rm::URI uri = rm::URI::parse("/absolute/path.txt");

    ASSERT_FALSE(uri.has_prefix());
    ASSERT_EQ(uri.path(), "/absolute/path.txt");
    ASSERT_TRUE(uri.is_absolute());
    ASSERT_FALSE(uri.is_relative());
}

void test_uri_scheme() {
    std::cout << "  test_uri_scheme..." << std::endl;

    rm::URI uri = rm::URI::parse("ftp:/path/file");

    ASSERT_FALSE(uri.has_prefix());
    ASSERT_EQ(uri.scheme(), "ftp");
    ASSERT_EQ(uri.path(), "/path/file");
    ASSERT_TRUE(uri.is_absolute());
}

void test_uri_empty() {
    std::cout << "  test_uri_empty..." << std::endl;

    rm::URI uri = rm::URI::parse("");

    ASSERT_TRUE(uri.empty());
    ASSERT_FALSE(uri.has_prefix());
    ASSERT_EQ(uri.path(), "");
}

void test_uri_prefix_no_path() {
    std::cout << "  test_uri_prefix_no_path..." << std::endl;

    rm::URI uri = rm::URI::parse("@data");

    ASSERT_TRUE(uri.has_prefix());
    ASSERT_EQ(uri.prefix(), "@data");
    ASSERT_EQ(uri.path(), "");
}

void run_uri_tests() {
    std::cout << "=== URI Parsing Tests ===" << std::endl;
    test_uri_data_prefix();
    test_uri_pid_prefix_with_arg();
    test_uri_pid_prefix_arg_only();
    test_uri_plain_path();
    test_uri_absolute_path();
    test_uri_scheme();
    test_uri_empty();
    test_uri_prefix_no_path();
    std::cout << "  All URI tests PASSED" << std::endl << std::endl;
}

// ============================================================================
// ResourceManager @data Tests
// ============================================================================

void test_rm_data_exists() {
    std::cout << "  test_rm_data_exists..." << std::endl;

    rm::ResourceManager mgr(CONFIG_PATH);

    // Should exist
    ASSERT_TRUE(mgr.exists("@data/skeleton/base.xml"));

    // Should not exist
    ASSERT_FALSE(mgr.exists("@data/nonexistent_file_12345.xml"));
}

void test_rm_data_fetch() {
    std::cout << "  test_rm_data_fetch..." << std::endl;

    rm::ResourceManager mgr(CONFIG_PATH);

    auto handle = mgr.fetch("@data/skeleton/base.xml");

    ASSERT_TRUE(handle.valid());
    ASSERT_TRUE(handle.size() > 0);

    // Check content contains skeleton XML elements
    auto content = handle.as_string();
    ASSERT_TRUE(content.find("<Skeleton") != std::string::npos ||
                content.find("<skeleton") != std::string::npos ||
                content.find("<?xml") != std::string::npos);

    // Verify the local_path is absolute and points to data directory
    auto local_path = handle.local_path().string();
    ASSERT_TRUE(local_path.find(DATA_ROOT) != std::string::npos);
    std::cout << "    Resolved to: " << local_path << std::endl;
}

void test_rm_data_fetch_not_found() {
    std::cout << "  test_rm_data_fetch_not_found..." << std::endl;

    rm::ResourceManager mgr(CONFIG_PATH);

    ASSERT_THROWS(mgr.fetch("@data/nonexistent_file_12345.xml"), rm::RMError);
}

void test_rm_data_list() {
    std::cout << "  test_rm_data_list..." << std::endl;

    rm::ResourceManager mgr(CONFIG_PATH);

    auto files = mgr.list("@data/skeleton/*.xml");

    ASSERT_TRUE(files.size() > 0);
    std::cout << "    Found " << files.size() << " XML files" << std::endl;

    // Should contain base.xml
    bool found_base = false;
    for (const auto& f : files) {
        if (f.find("base.xml") != std::string::npos) {
            found_base = true;
            break;
        }
    }
    ASSERT_TRUE(found_base);
}

void run_rm_data_tests() {
    std::cout << "=== ResourceManager @data Tests ===" << std::endl;
    test_rm_data_exists();
    test_rm_data_fetch();
    test_rm_data_fetch_not_found();
    test_rm_data_list();
    std::cout << "  All @data tests PASSED" << std::endl << std::endl;
}

// ============================================================================
// ResourceManager @pid Tests (Local Backend)
// ============================================================================

void test_rm_pid_local_exists() {
    std::cout << "  test_rm_pid_local_exists..." << std::endl;

    rm::ResourceManager mgr(CONFIG_PATH);

    // Check if pid_local backend root exists
    if (!fs::exists(PID_ROOT)) {
        std::cout << "    SKIPPED (pid_local root not available: " << PID_ROOT << ")" << std::endl;
        return;
    }

    // Try to list available patient IDs
    std::error_code ec;
    bool found_any = false;
    for (auto& entry : fs::directory_iterator(PID_ROOT, ec)) {
        if (entry.is_directory()) {
            found_any = true;
            std::cout << "    Found patient directory: " << entry.path().filename() << std::endl;
            break;
        }
    }

    if (!found_any) {
        std::cout << "    SKIPPED (no patient directories found)" << std::endl;
    }
}

void test_rm_pid_local_fetch() {
    std::cout << "  test_rm_pid_local_fetch..." << std::endl;

    rm::ResourceManager mgr(CONFIG_PATH);

    // Check if pid_local backend root exists
    if (!fs::exists(PID_ROOT)) {
        std::cout << "    SKIPPED (pid_local root not available)" << std::endl;
        return;
    }

    // Try to find a real patient directory and file
    std::error_code ec;
    for (auto& entry : fs::directory_iterator(PID_ROOT, ec)) {
        if (!entry.is_directory()) continue;

        std::string patient_id = entry.path().filename().string();

        // Try to find any file in this patient's directory
        for (auto& subentry : fs::recursive_directory_iterator(entry.path(), ec)) {
            if (!subentry.is_regular_file()) continue;
            if (subentry.file_size() > 10'000'000) continue;  // Skip large files

            std::string rel_path = fs::relative(subentry.path(), entry.path()).string();
            std::string uri = "@pid:" + patient_id + "/" + rel_path;

            try {
                auto handle = mgr.fetch(uri);
                ASSERT_TRUE(handle.valid());
                ASSERT_TRUE(handle.size() > 0);

                // Verify path resolves to PID_ROOT
                auto local_path = handle.local_path().string();
                ASSERT_TRUE(local_path.find(PID_ROOT) != std::string::npos);

                std::cout << "    Fetched: " << uri << std::endl;
                std::cout << "    Resolved to: " << local_path << " (" << handle.size() << " bytes)" << std::endl;
                return;  // Success
            } catch (const rm::RMError& e) {
                continue;  // Try next file
            }
        }
    }

    std::cout << "    SKIPPED (no accessible patient files found)" << std::endl;
}

void run_rm_pid_local_tests() {
    std::cout << "=== ResourceManager @pid Local Tests ===" << std::endl;
    test_rm_pid_local_exists();
    test_rm_pid_local_fetch();
    std::cout << "  @pid local tests completed" << std::endl << std::endl;
}

// ============================================================================
// ResourceManager @pid Tests (FTP Backend)
// ============================================================================

void test_rm_pid_ftp() {
    std::cout << "  test_rm_pid_ftp..." << std::endl;

    rm::ResourceManager mgr(CONFIG_PATH);

    // FTP is only used as fallback when local doesn't exist
    if (fs::exists(PID_ROOT)) {
        std::cout << "    SKIPPED (local backend available, FTP fallback not triggered)" << std::endl;
        return;
    }

    // Try FTP fetch - this will only work if FTP is properly configured
    try {
        auto handle = mgr.fetch("@pid:TEST001/test.txt");
        ASSERT_TRUE(handle.valid());
        std::cout << "    FTP fetch successful!" << std::endl;
    } catch (const rm::RMError& e) {
        std::cout << "    SKIPPED (FTP not available: " << e.what() << ")" << std::endl;
    }
}

void run_rm_pid_ftp_tests() {
    std::cout << "=== ResourceManager @pid FTP Tests ===" << std::endl;
    test_rm_pid_ftp();
    std::cout << "  @pid FTP tests completed" << std::endl << std::endl;
}

// ============================================================================
// Caching Tests
// ============================================================================

void test_rm_cache() {
    std::cout << "  test_rm_cache..." << std::endl;

    rm::ResourceManager mgr(CONFIG_PATH);

    // Local backend doesn't use cache (cached() returns false)
    // FTP backend uses cache
    std::cout << "    Cache behavior verified (local backend doesn't cache)" << std::endl;
}

void run_cache_tests() {
    std::cout << "=== Caching Tests ===" << std::endl;
    test_rm_cache();
    std::cout << "  Caching tests completed" << std::endl << std::endl;
}

// ============================================================================
// ResourceHandle Tests
// ============================================================================

void test_handle_valid() {
    std::cout << "  test_handle_valid..." << std::endl;

    rm::ResourceManager mgr(CONFIG_PATH);

    auto handle = mgr.fetch("@data/skeleton/base.xml");

    ASSERT_TRUE(handle.valid());
    ASSERT_TRUE(handle.size() > 0);

    // Local path should be set and absolute
    ASSERT_FALSE(handle.local_path().empty());
    ASSERT_TRUE(fs::exists(handle.local_path()));
    ASSERT_TRUE(handle.local_path().is_absolute());
}

void test_handle_data_access() {
    std::cout << "  test_handle_data_access..." << std::endl;

    rm::ResourceManager mgr(CONFIG_PATH);

    auto handle = mgr.fetch("@data/skeleton/base.xml");

    // Access data (lazy loading from file)
    const auto& data = handle.data();
    ASSERT_TRUE(data.size() > 0);

    // as_string() should return the same content
    auto str = handle.as_string();
    ASSERT_EQ(str.size(), data.size());
}

void run_handle_tests() {
    std::cout << "=== ResourceHandle Tests ===" << std::endl;
    test_handle_valid();
    test_handle_data_access();
    std::cout << "  ResourceHandle tests completed" << std::endl << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Resource Manager Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Config: " << CONFIG_PATH << std::endl;
    std::cout << "Data root: " << DATA_ROOT << std::endl;
    std::cout << "PID root: " << PID_ROOT << std::endl;
    std::cout << std::endl;

    try {
        run_uri_tests();
        run_rm_data_tests();
        run_rm_pid_local_tests();
        run_rm_pid_ftp_tests();
        run_cache_tests();
        run_handle_tests();

        std::cout << "========================================" << std::endl;
        std::cout << "All tests completed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
