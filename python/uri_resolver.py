"""
Python URI Resolver for BidirectionalGaitNet
Provides path resolution compatible with the C++ UriResolver
"""

import os
import sys
import re
from pathlib import Path
from ftplib import FTP
import yaml
from python.log_config import log_verbose


class URIResolver:
    """Python version of the C++ URIResolver for consistent path handling"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(URIResolver, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.scheme_roots = {}
            self.ftp_credentials_cache = {}
            self._initialized = False
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        return cls()
    
    def initialize(self):
        """Initialize the resolver with default schemes"""
        if self._initialized:
            return
            
        # Get project root from environment or fallback detection
        project_root = self._get_project_root()
        data_root = os.path.join(project_root, "data")
        
        self.register_scheme("data", data_root)
        self._initialized = True
        log_verbose(f"[Python] URIResolver initialized with data root: {data_root}")
    
    def resolve(self, uri):
        """
        Resolve a URI to an absolute path
        Supports @ftp:host/path, @scheme/path, scheme:path, */path, and ../data/path formats
        """
        if not self.is_uri(uri):
            # Handle */filename pattern
            if uri.startswith("*/") and len(uri) > 2:
                filename = uri[2:]  # Remove */
                if "data" in self.scheme_roots:
                    return os.path.join(self.scheme_roots["data"], filename)

            # Handle ../data/ pattern (backwards compatibility)
            if uri.startswith("../data/"):
                filename = uri[8:]  # Remove "../data/"
                if "data" in self.scheme_roots:
                    resolved_path = os.path.join(self.scheme_roots["data"], filename)
                    print(f"Python URIResolver: Backwards compatibility - resolving {uri} to {resolved_path}")
                    return resolved_path

            # Handle ./rollout/ pattern (project relative)
            if uri.startswith("./"):
                project_root = self._get_project_root()
                relative_path = uri[2:]  # Remove "./"
                resolved_path = os.path.join(project_root, relative_path)
                return resolved_path

            return uri  # Return as-is if not a recognized pattern

        # Parse URI - Check for @ftp:host/path format
        if uri.startswith("@ftp:"):
            # @ftp:host/path format
            host_path = uri[5:]  # Remove "@ftp:"
            parts = host_path.split("/", 1)
            if len(parts) != 2:
                print(f"Error: Invalid FTP URI format (missing path): {uri}")
                sys.exit(1)
            host, path = parts
            return self._resolve_ftp(host, "/" + path)

        # Parse URI for other schemes
        if uri.startswith("@"):
            # @scheme/path format
            parts = uri[1:].split("/", 1)
            if len(parts) != 2:
                print(f"Warning: Invalid URI format: {uri}")
                return uri
            scheme, relative_path = parts
        else:
            # scheme:path format
            if ":" not in uri:
                print(f"Warning: Invalid URI format: {uri}")
                return uri
            scheme, relative_path = uri.split(":", 1)

        # Resolve scheme
        if scheme not in self.scheme_roots:
            print(f"Warning: Unknown URI scheme: {scheme}")
            return uri

        return os.path.join(self.scheme_roots[scheme], relative_path)
    
    def is_uri(self, path):
        """Check if a string is a URI"""
        if not path:
            return False
            
        # Check for @scheme/path format
        if path.startswith("@"):
            return True
            
        # Check for scheme:path format
        if ":" in path:
            colon_pos = path.find(":")
            if colon_pos > 0:
                # Make sure it's not a Windows drive letter
                if colon_pos == 1 and len(path) > 2 and path[2] in ["\\", "/"]:
                    return False
                # Make sure scheme doesn't contain path separators
                scheme = path[:colon_pos]
                return "/" not in scheme and "\\" not in scheme
                
        return False
    
    def register_scheme(self, scheme, root_path):
        """Register a new scheme with its root path"""
        self.scheme_roots[scheme] = root_path
    
    def _get_project_root(self):
        """Get the project root directory"""
        # Try environment variable first
        if "PROJECT_ROOT" in os.environ:
            return os.environ["PROJECT_ROOT"]
            
        # Try to detect project root by looking for characteristic files
        current_dir = Path(__file__).parent.absolute()
        
        # Look for project root indicators
        for parent in [current_dir] + list(current_dir.parents):
            if (parent / "CMakeLists.txt").exists() and (parent / "data").exists():
                return str(parent)
                
        # Fallback: assume project is parent of python directory
        return str(current_dir.parent)

    def _get_temp_dir(self):
        """Get the temp directory for FTP downloads"""
        project_root = self._get_project_root()
        temp_dir = os.path.join(project_root, ".temp")

        # Create .temp directory if it doesn't exist
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)

        return temp_dir

    def _generate_temp_filepath(self, host, remote_path):
        """Generate safe temp file path for FTP download"""
        temp_dir = self._get_temp_dir()

        # Convert remote path to safe filename: /path/to/file.xml -> path_to_file.xml
        safe_filename = remote_path.replace("/", "_")
        if safe_filename.startswith("_"):
            safe_filename = safe_filename[1:]

        # Prepend host to avoid conflicts: ftp_host_path_to_file.xml
        full_filename = f"ftp_{host}_{safe_filename}"

        return os.path.join(temp_dir, full_filename)

    def _load_ftp_credentials(self, host):
        """Load FTP credentials from data/secret.yaml"""
        # Check cache first
        if host in self.ftp_credentials_cache:
            return self.ftp_credentials_cache[host]

        # Load from secret.yaml
        project_root = self._get_project_root()
        secret_path = os.path.join(project_root, "data", "secret.yaml")

        try:
            with open(secret_path, 'r') as f:
                config = yaml.safe_load(f)

            if host in config:
                creds = {
                    'ip': config[host]['ip'],
                    'username': config[host]['username'],
                    'password': config[host]['password'],
                    'port': config[host]['port']
                }
                # Cache for future use
                self.ftp_credentials_cache[host] = creds
                return creds
            else:
                print(f"Error: FTP host '{host}' not found in secret.yaml")
                sys.exit(1)
        except Exception as e:
            print(f"Error loading FTP credentials: {e}")
            sys.exit(1)

    def _list_ftp_directory(self, host, path, creds, directories_only=True):
        """List directories/files in an FTP path"""
        try:
            ftp = FTP()
            ftp.connect(creds['ip'], creds['port'])
            ftp.login(creds['username'], creds['password'])

            # Change to directory
            if path and path != '/':
                ftp.cwd(path)

            # List directory
            lines = []
            ftp.retrlines('LIST', lines.append)
            ftp.quit()

            # Parse names from listing
            directories = []
            for line in lines:
                # Entries start with 'd' (directory) or '-' (file)
                is_directory = line.startswith('d')
                is_file = line.startswith('-')

                if is_directory or (not directories_only and is_file):
                    # Extract last field (name)
                    parts = line.split()
                    if len(parts) >= 9:
                        name = parts[-1]
                        if name not in ['.', '..']:
                            directories.append(name)

            # Sort in descending order (latest first)
            directories.sort(reverse=True)
            return directories

        except Exception as e:
            print(f"Error: FTP directory listing failed for path '{path}': {e}")
            print(f"       Host: {creds['ip']}:{creds['port']}")
            sys.exit(1)

    def _resolve_wildcard_path(self, host, path, creds):
        """Resolve #n and * wildcards in FTP path - segment by segment"""
        # Split path by / and process each segment
        segments = [s for s in path.split('/') if s]
        resolved_path = ""

        for seg_idx, seg in enumerate(segments):
            is_last_segment = (seg_idx == len(segments) - 1)
            # Check if segment contains #n wildcard
            if '#' in seg:
                match = re.search(r'#(\d+)', seg)
                if not match:
                    print(f"Error: Invalid #n format in segment: {seg}")
                    sys.exit(1)

                depth = int(match.group(1))

                # Descend n levels from current path (always directories only)
                current_path = resolved_path
                for i in range(depth):
                    dirs = self._list_ftp_directory(host, current_path, creds, directories_only=True)
                    if not dirs:
                        print(f"Error: No directories at '{current_path}' (depth {i+1}/{depth})")
                        sys.exit(1)

                    resolved_path = f"{resolved_path}/{dirs[0]}"
                    current_path = resolved_path
                    print(f"FTP: #{i+1} → {dirs[0]}")

            # Check if segment contains * wildcard
            elif '*' in seg:
                star_pos = seg.find('*')
                prefix = seg[:star_pos]
                suffix = seg[star_pos + 1:]

                # List current directory and find match
                # If this is the last segment, include files; otherwise directories only
                print(f"FTP: Listing for pattern '{prefix}*{suffix}' at: {resolved_path} (last={is_last_segment})")
                dirs = self._list_ftp_directory(host, resolved_path, creds, directories_only=not is_last_segment)

                matched = None
                for dir_name in dirs:
                    if dir_name.startswith(prefix):
                        if not suffix or suffix in dir_name[len(prefix):]:
                            matched = dir_name
                            break

                if not matched:
                    print(f"Error: No match for '{prefix}*{suffix}' at '{resolved_path}'")
                    print(f"       Available: {', '.join(dirs[:5])}")
                    sys.exit(1)

                resolved_path = f"{resolved_path}/{matched}"
                print(f"FTP: {prefix}*{suffix} → {matched}")

            else:
                # Regular segment, just append
                resolved_path = f"{resolved_path}/{seg}"

        return resolved_path

    def _download_ftp_file(self, host, remote_path, dest_path, creds):
        """Download file from FTP server"""
        try:
            ftp = FTP()
            ftp.connect(creds['ip'], creds['port'])
            ftp.login(creds['username'], creds['password'])

            # Download file (will fail if file doesn't exist)
            with open(dest_path, 'wb') as f:
                ftp.retrbinary(f'RETR {remote_path}', f.write)

            ftp.quit()
            print(f"FTP: Downloaded {remote_path} to {dest_path}")
            return dest_path
        except Exception as e:
            print(f"Error: FTP download failed: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)  # Clean up partial file
            sys.exit(1)

    def _resolve_ftp(self, host, path):
        """Resolve FTP URI to local temp file path"""
        # Load credentials for the host
        creds = self._load_ftp_credentials(host)

        # Resolve wildcards if present (# or *)
        resolved_path = path
        if '#' in path or '*' in path:
            resolved_path = self._resolve_wildcard_path(host, path, creds)
            print(f"FTP: Wildcard path {path} resolved to {resolved_path}")

        # Generate temp file path
        temp_filepath = self._generate_temp_filepath(host, resolved_path)

        # Check if file already exists in temp (cached)
        if os.path.exists(temp_filepath):
            print(f"FTP: Using cached file: {temp_filepath}")
            return temp_filepath

        # Download the file
        return self._download_ftp_file(host, resolved_path, temp_filepath, creds)


# Global instance for easy access
uri_resolver = URIResolver.get_instance()


def resolve_path(uri):
    """Convenience function to resolve a path"""
    uri_resolver.initialize()
    return uri_resolver.resolve(uri)


def ensure_directory_exists(path):
    """Ensure directory exists, create if necessary"""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")