#pragma once
#include "rm/manager.hpp"
#include <string>

namespace rm {

// Thread-safe singleton accessor (Meyer's singleton)
// C++11 guarantees thread-safe initialization of static locals
ResourceManager& getManager();

// Convenience function - replaces URIResolver::getInstance().resolve()
// Returns the original string if it's not a URI (doesn't start with @)
std::string resolve(const std::string& uri);

} // namespace rm
