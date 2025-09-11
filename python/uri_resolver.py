"""
Python URI Resolver for BidirectionalGaitNet
Provides path resolution compatible with the C++ UriResolver
"""

import os
from pathlib import Path


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
        print(f"Python URIResolver initialized with data root: {data_root}")
    
    def resolve(self, uri):
        """
        Resolve a URI to an absolute path
        Supports @scheme/path, scheme:path, */path, and ../data/path formats
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
        
        # Parse URI
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