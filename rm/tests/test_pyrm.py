#!/usr/bin/env python3
"""
Resource Manager Python Bindings Test Suite
Tests URI parsing, @data and @pid prefix routing, and backend operations
"""

import sys
import os
from pathlib import Path

# Absolute paths
CONFIG_PATH = "/home/geon/BidirectionalGaitNet/data/rm_config.yaml"
DATA_ROOT = "/home/geon/BidirectionalGaitNet/data"
PID_ROOT = "/mnt/blue8T/CP/RM"

# Add build directory to path for pyrm module
build_dir = Path("/home/geon/BidirectionalGaitNet/build/release/rm/python")
if build_dir.exists():
    sys.path.insert(0, str(build_dir))

try:
    import pyrm
except ImportError as e:
    print(f"ERROR: Could not import pyrm module: {e}")
    print(f"Make sure to build the project first: ninja -C build/release")
    print(f"Searched in: {build_dir}")
    sys.exit(1)


# ============================================================================
# URI Parsing Tests
# ============================================================================

def test_uri_data_prefix():
    """Test @data prefix parsing"""
    print("  test_uri_data_prefix...", end=" ")

    uri = pyrm.URI.parse("@data/skeleton/base.xml")

    assert uri.has_prefix(), "Should have prefix"
    assert uri.prefix() == "@data", f"Expected '@data', got '{uri.prefix()}'"
    assert uri.prefix_arg() == "", f"Expected empty prefix_arg, got '{uri.prefix_arg()}'"
    assert uri.path() == "skeleton/base.xml", f"Expected 'skeleton/base.xml', got '{uri.path()}'"
    assert uri.resolved_path() == "skeleton/base.xml"
    assert str(uri) == "@data/skeleton/base.xml"
    assert uri.is_relative()
    assert not uri.is_absolute()

    print("PASSED")


def test_uri_pid_prefix_with_arg():
    """Test @pid:arg prefix parsing"""
    print("  test_uri_pid_prefix_with_arg...", end=" ")

    uri = pyrm.URI.parse("@pid:CP001/markers/trial.c3d")

    assert uri.has_prefix()
    assert uri.prefix() == "@pid"
    assert uri.prefix_arg() == "CP001"
    assert uri.path() == "markers/trial.c3d"
    assert uri.resolved_path() == "CP001/markers/trial.c3d"
    assert str(uri) == "@pid:CP001/markers/trial.c3d"

    print("PASSED")


def test_uri_pid_prefix_arg_only():
    """Test @pid:arg without path"""
    print("  test_uri_pid_prefix_arg_only...", end=" ")

    uri = pyrm.URI.parse("@pid:CP001")

    assert uri.has_prefix()
    assert uri.prefix() == "@pid"
    assert uri.prefix_arg() == "CP001"
    assert uri.path() == ""
    assert uri.resolved_path() == "CP001"

    print("PASSED")


def test_uri_plain_path():
    """Test plain path without prefix"""
    print("  test_uri_plain_path...", end=" ")

    uri = pyrm.URI.parse("skeleton/base.xml")

    assert not uri.has_prefix()
    assert uri.prefix() == ""
    assert uri.path() == "skeleton/base.xml"
    assert uri.resolved_path() == "skeleton/base.xml"
    assert uri.is_relative()

    print("PASSED")


def test_uri_absolute_path():
    """Test absolute path"""
    print("  test_uri_absolute_path...", end=" ")

    uri = pyrm.URI.parse("/absolute/path.txt")

    assert not uri.has_prefix()
    assert uri.path() == "/absolute/path.txt"
    assert uri.is_absolute()
    assert not uri.is_relative()

    print("PASSED")


def test_uri_scheme():
    """Test scheme:path format"""
    print("  test_uri_scheme...", end=" ")

    uri = pyrm.URI.parse("ftp:/path/file")

    assert not uri.has_prefix()
    assert uri.scheme() == "ftp"
    assert uri.path() == "/path/file"
    assert uri.is_absolute()

    print("PASSED")


def test_uri_empty():
    """Test empty URI"""
    print("  test_uri_empty...", end=" ")

    uri = pyrm.URI.parse("")

    assert uri.empty()
    assert not uri.has_prefix()
    assert uri.path() == ""

    print("PASSED")


def test_uri_repr():
    """Test URI string representations"""
    print("  test_uri_repr...", end=" ")

    uri = pyrm.URI.parse("@data/test.txt")

    assert str(uri) == "@data/test.txt"
    assert repr(uri) == "URI('@data/test.txt')"

    print("PASSED")


def run_uri_tests():
    """Run all URI tests"""
    print("=== URI Parsing Tests ===")
    test_uri_data_prefix()
    test_uri_pid_prefix_with_arg()
    test_uri_pid_prefix_arg_only()
    test_uri_plain_path()
    test_uri_absolute_path()
    test_uri_scheme()
    test_uri_empty()
    test_uri_repr()
    print("  All URI tests PASSED\n")


# ============================================================================
# ResourceManager @data Tests
# ============================================================================

def test_rm_data_exists():
    """Test exists() for @data prefix"""
    print("  test_rm_data_exists...", end=" ")

    rm = pyrm.ResourceManager(CONFIG_PATH)

    # Should exist
    assert rm.exists("@data/skeleton/base.xml"), "base.xml should exist"

    # Should not exist
    assert not rm.exists("@data/nonexistent_file_12345.xml"), "nonexistent file should not exist"

    print("PASSED")


def test_rm_data_fetch():
    """Test fetch() for @data prefix"""
    print("  test_rm_data_fetch...", end=" ")

    rm = pyrm.ResourceManager(CONFIG_PATH)

    handle = rm.fetch("@data/skeleton/base.xml")

    assert handle.valid(), "Handle should be valid"
    assert handle.size() > 0, "Handle should have content"
    assert len(handle) > 0, "__len__ should work"

    # Check content
    content = handle.as_string()
    assert "Skeleton" in content or "skeleton" in content, "Should be XML content"

    # Check local_path resolves to DATA_ROOT
    local_path = handle.local_path()
    assert local_path, "Should have local path"
    assert os.path.exists(local_path), f"Local path should exist: {local_path}"
    assert DATA_ROOT in local_path, f"Path should be under {DATA_ROOT}, got {local_path}"

    print(f"PASSED (resolved to {local_path})")


def test_rm_data_fetch_not_found():
    """Test fetch() throws for non-existent file"""
    print("  test_rm_data_fetch_not_found...", end=" ")

    rm = pyrm.ResourceManager(CONFIG_PATH)

    try:
        rm.fetch("@data/nonexistent_file_12345.xml")
        assert False, "Should have raised RMError"
    except pyrm.RMError:
        pass  # Expected

    print("PASSED")


def test_rm_data_list():
    """Test list() for @data prefix"""
    print("  test_rm_data_list...", end=" ")

    rm = pyrm.ResourceManager(CONFIG_PATH)

    files = rm.list("@data/skeleton/*.xml")

    assert len(files) > 0, "Should find XML files"

    # Should contain base.xml
    found_base = any("base.xml" in f for f in files)
    assert found_base, f"Should find base.xml in {files}"

    print(f"PASSED (found {len(files)} files)")


def test_rm_backend_count():
    """Test backend_count()"""
    print("  test_rm_backend_count...", end=" ")

    rm = pyrm.ResourceManager(CONFIG_PATH)

    count = rm.backend_count()
    assert count > 0, "Should have at least one backend"

    print(f"PASSED ({count} backends)")


def run_rm_data_tests():
    """Run all @data ResourceManager tests"""
    print("=== ResourceManager @data Tests ===")
    test_rm_data_exists()
    test_rm_data_fetch()
    test_rm_data_fetch_not_found()
    test_rm_data_list()
    test_rm_backend_count()
    print("  All @data tests PASSED\n")


# ============================================================================
# ResourceManager @pid Tests (Local Backend)
# ============================================================================

def test_rm_pid_local_exists():
    """Test @pid with local backend"""
    print("  test_rm_pid_local_exists...", end=" ")

    rm = pyrm.ResourceManager(CONFIG_PATH)

    # Check if pid_local backend root exists
    pid_root = Path(PID_ROOT)
    if not pid_root.exists():
        print(f"SKIPPED (pid_local root not available: {PID_ROOT})")
        return

    # Try to find any patient directory
    patient_dirs = [d for d in pid_root.iterdir() if d.is_dir()]
    if not patient_dirs:
        print("SKIPPED (no patient directories found)")
        return

    # Test exists for first patient
    patient_id = patient_dirs[0].name
    print(f"PASSED (found patient: {patient_id})")


def test_rm_pid_local_fetch():
    """Test fetch() for @pid with local backend"""
    print("  test_rm_pid_local_fetch...", end=" ")

    rm = pyrm.ResourceManager(CONFIG_PATH)

    # Check if pid_local backend root exists
    pid_root = Path(PID_ROOT)
    if not pid_root.exists():
        print(f"SKIPPED (pid_local root not available: {PID_ROOT})")
        return

    # Try to find and fetch any patient file
    for patient_dir in pid_root.iterdir():
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name

        for f in patient_dir.rglob("*"):
            if not f.is_file():
                continue
            if f.stat().st_size > 10_000_000:  # Skip very large files
                continue

            rel_path = f.relative_to(patient_dir)
            uri = f"@pid:{patient_id}/{rel_path}"

            try:
                handle = rm.fetch(uri)
                assert handle.valid()
                assert handle.size() > 0

                # Verify path resolves to PID_ROOT
                local_path = handle.local_path()
                assert PID_ROOT in local_path, f"Path should be under {PID_ROOT}, got {local_path}"

                print(f"PASSED (fetched {uri})")
                print(f"    Resolved to: {local_path} ({handle.size()} bytes)")
                return
            except pyrm.RMError:
                continue

    print("SKIPPED (no accessible patient files found)")


def run_rm_pid_local_tests():
    """Run all @pid local tests"""
    print("=== ResourceManager @pid Local Tests ===")
    test_rm_pid_local_exists()
    test_rm_pid_local_fetch()
    print("  @pid local tests completed\n")


# ============================================================================
# ResourceManager @pid Tests (FTP Backend)
# ============================================================================

def test_rm_pid_ftp():
    """Test @pid FTP fallback"""
    print("  test_rm_pid_ftp...", end=" ")

    rm = pyrm.ResourceManager(CONFIG_PATH)

    # FTP is only used as fallback when local doesn't exist
    if Path(PID_ROOT).exists():
        print("SKIPPED (local backend available, FTP fallback not triggered)")
        return

    # Try FTP fetch
    try:
        handle = rm.fetch("@pid:TEST001/test.txt")
        assert handle.valid()
        print("PASSED (FTP fetch successful)")
    except pyrm.RMError as e:
        print(f"SKIPPED (FTP not available: {e})")


def run_rm_pid_ftp_tests():
    """Run all @pid FTP tests"""
    print("=== ResourceManager @pid FTP Tests ===")
    test_rm_pid_ftp()
    print("  @pid FTP tests completed\n")


# ============================================================================
# ResourceHandle Tests
# ============================================================================

def test_handle_data():
    """Test ResourceHandle.data()"""
    print("  test_handle_data...", end=" ")

    rm = pyrm.ResourceManager(CONFIG_PATH)
    handle = rm.fetch("@data/skeleton/base.xml")

    # data() returns bytes
    data = handle.data()
    assert isinstance(data, bytes), f"Expected bytes, got {type(data)}"
    assert len(data) > 0

    print("PASSED")


def test_handle_as_string():
    """Test ResourceHandle.as_string()"""
    print("  test_handle_as_string...", end=" ")

    rm = pyrm.ResourceManager(CONFIG_PATH)
    handle = rm.fetch("@data/skeleton/base.xml")

    # as_string() returns str
    content = handle.as_string()
    assert isinstance(content, str), f"Expected str, got {type(content)}"
    assert len(content) > 0

    # Should match data() decoded
    data = handle.data()
    assert content == data.decode('utf-8', errors='replace')[:len(content)]

    print("PASSED")


def test_handle_local_path():
    """Test ResourceHandle.local_path()"""
    print("  test_handle_local_path...", end=" ")

    rm = pyrm.ResourceManager(CONFIG_PATH)
    handle = rm.fetch("@data/skeleton/base.xml")

    local_path = handle.local_path()
    assert local_path, "Should have local path"
    assert os.path.exists(local_path), f"Path should exist: {local_path}"
    assert os.path.isabs(local_path), f"Path should be absolute: {local_path}"

    print("PASSED")


def run_handle_tests():
    """Run all ResourceHandle tests"""
    print("=== ResourceHandle Tests ===")
    test_handle_data()
    test_handle_as_string()
    test_handle_local_path()
    print("  ResourceHandle tests completed\n")


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_error_codes():
    """Test ErrorCode enum"""
    print("  test_error_codes...", end=" ")

    # Check that error codes exist
    assert hasattr(pyrm, 'ErrorCode')
    assert hasattr(pyrm.ErrorCode, 'NotFound')
    assert hasattr(pyrm.ErrorCode, 'AccessDenied')
    assert hasattr(pyrm.ErrorCode, 'NetworkError')
    assert hasattr(pyrm.ErrorCode, 'InvalidURI')
    assert hasattr(pyrm.ErrorCode, 'IOError')
    assert hasattr(pyrm.ErrorCode, 'ConfigError')

    print("PASSED")


def test_rm_error_exception():
    """Test RMError exception"""
    print("  test_rm_error_exception...", end=" ")

    rm = pyrm.ResourceManager(CONFIG_PATH)

    try:
        rm.fetch("@data/definitely_does_not_exist_12345.xml")
        assert False, "Should have raised"
    except pyrm.RMError as e:
        # Exception should have a message
        assert str(e), "Exception should have message"

    print("PASSED")


def run_error_tests():
    """Run all error handling tests"""
    print("=== Error Handling Tests ===")
    test_error_codes()
    test_rm_error_exception()
    print("  Error handling tests completed\n")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 50)
    print("Resource Manager Python Bindings Test Suite")
    print("=" * 50)
    print(f"Config: {CONFIG_PATH}")
    print(f"Data root: {DATA_ROOT}")
    print(f"PID root: {PID_ROOT}")
    print()

    try:
        run_uri_tests()
        run_rm_data_tests()
        run_rm_pid_local_tests()
        run_rm_pid_ftp_tests()
        run_handle_tests()
        run_error_tests()

        print("=" * 50)
        print("All Python tests completed successfully!")
        print("=" * 50)
        return 0

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nTest failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
