#pragma once

#include <exception>
#include <string>

namespace rm {

enum class ErrorCode {
    NotFound,       // Resource doesn't exist
    AccessDenied,   // Permission/auth failure
    NetworkError,   // Transport-level failure (FTP connection, etc.)
    InvalidURI,     // Malformed URI
    IOError,        // Local filesystem error
    ConfigError     // YAML config parsing error
};

inline const char* error_code_to_string(ErrorCode code) {
    switch (code) {
        case ErrorCode::NotFound:     return "NotFound";
        case ErrorCode::AccessDenied: return "AccessDenied";
        case ErrorCode::NetworkError: return "NetworkError";
        case ErrorCode::InvalidURI:   return "InvalidURI";
        case ErrorCode::IOError:      return "IOError";
        case ErrorCode::ConfigError:  return "ConfigError";
        default:                      return "Unknown";
    }
}

class RMError : public std::exception {
public:
    RMError(ErrorCode code, const std::string& message)
        : code_(code), message_(message), uri_() {
        build_what();
    }

    RMError(ErrorCode code, const std::string& uri, const std::string& message)
        : code_(code), message_(message), uri_(uri) {
        build_what();
    }

    ErrorCode code() const noexcept { return code_; }
    const char* what() const noexcept override { return what_.c_str(); }
    const std::string& message() const noexcept { return message_; }
    const std::string& uri() const noexcept { return uri_; }

private:
    void build_what() {
        what_ = std::string("[rm::") + error_code_to_string(code_) + "] " + message_;
        if (!uri_.empty()) {
            what_ += " (uri: " + uri_ + ")";
        }
    }

    ErrorCode code_;
    std::string message_;
    std::string uri_;
    std::string what_;
};

} // namespace rm
