#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "rm/rm.hpp"
#include <set>

namespace py = pybind11;

PYBIND11_MODULE(pyrm, m) {
    m.doc() = "Resource Manager Core - Python bindings";

    // Error codes
    py::enum_<rm::ErrorCode>(m, "ErrorCode")
        .value("NotFound", rm::ErrorCode::NotFound)
        .value("AccessDenied", rm::ErrorCode::AccessDenied)
        .value("NetworkError", rm::ErrorCode::NetworkError)
        .value("InvalidURI", rm::ErrorCode::InvalidURI)
        .value("IOError", rm::ErrorCode::IOError)
        .value("ConfigError", rm::ErrorCode::ConfigError)
        .export_values();

    // RMError exception
    py::register_exception<rm::RMError>(m, "RMError");

    // URI class
    py::class_<rm::URI>(m, "URI")
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def_static("parse", &rm::URI::parse)
        .def("is_relative", &rm::URI::is_relative)
        .def("is_absolute", &rm::URI::is_absolute)
        .def("scheme", &rm::URI::scheme)
        .def("path", &rm::URI::path)
        .def("has_prefix", &rm::URI::has_prefix)
        .def("prefix", &rm::URI::prefix)
        .def("prefix_arg", &rm::URI::prefix_arg)
        .def("resolved_path", &rm::URI::resolved_path)
        .def("to_string", &rm::URI::to_string)
        .def("empty", &rm::URI::empty)
        .def("__str__", &rm::URI::to_string)
        .def("__repr__", [](const rm::URI& uri) {
            return "URI('" + uri.to_string() + "')";
        });

    // ResourceHandle class
    py::class_<rm::ResourceHandle>(m, "ResourceHandle")
        .def("data", [](const rm::ResourceHandle& h) {
            const auto& d = h.data();
            return py::bytes(reinterpret_cast<const char*>(d.data()), d.size());
        })
        .def("local_path", [](const rm::ResourceHandle& h) {
            return h.local_path().string();
        })
        .def("as_string", [](const rm::ResourceHandle& h) {
            auto sv = h.as_string();
            return std::string(sv);
        })
        .def("valid", &rm::ResourceHandle::valid)
        .def("size", &rm::ResourceHandle::size)
        .def("__len__", &rm::ResourceHandle::size);

    // ResourceManager class
    py::class_<rm::ResourceManager>(m, "ResourceManager")
        .def(py::init<const std::string&>(), py::arg("config_path"))
        // exists: accept string or URI
        .def("exists", [](rm::ResourceManager& mgr, const std::string& uri) {
            return mgr.exists(uri);
        }, py::arg("uri"))
        .def("exists", [](rm::ResourceManager& mgr, const rm::URI& uri) {
            return mgr.exists(uri.to_string());
        }, py::arg("uri"))
        // fetch: accept string or URI
        .def("fetch", [](rm::ResourceManager& mgr, const std::string& uri) {
            return mgr.fetch(uri);
        }, py::arg("uri"), py::return_value_policy::move)
        .def("fetch", [](rm::ResourceManager& mgr, const rm::URI& uri) {
            return mgr.fetch(uri.to_string());
        }, py::arg("uri"), py::return_value_policy::move)
        .def("list", &rm::ResourceManager::list, py::arg("pattern"))
        .def("backend_count", &rm::ResourceManager::backend_count)
        // resolve: accept string or URI
        .def("resolve", [](rm::ResourceManager& mgr, const std::string& uri) {
            return mgr.resolve(uri).string();
        }, py::arg("uri"), "Resolve URI to full filesystem path")
        .def("resolve", [](rm::ResourceManager& mgr, const rm::URI& uri) {
            return mgr.resolve(uri.to_string()).string();
        }, py::arg("uri"), "Resolve URI to full filesystem path")
        // resolve_backend_names: accept string or URI
        .def("resolve_backend_names", [](rm::ResourceManager& mgr, const std::string& uri) {
            return mgr.resolve_backend_names(uri);
        }, py::arg("uri"), "Get backend names for a URI")
        .def("resolve_backend_names", [](rm::ResourceManager& mgr, const rm::URI& uri) {
            return mgr.resolve_backend_names(uri.to_string());
        }, py::arg("uri"), "Get backend names for a URI")
        // resolve_dir: resolve @pid: URI to directory path
        .def("resolve_dir", [](rm::ResourceManager& mgr, const std::string& uri) {
            auto path = mgr.resolveDir(uri);
            if (path.empty()) {
                throw rm::RMError(rm::ErrorCode::NotFound, "Directory not found: " + uri);
            }
            return path.string();
        }, py::arg("uri"), "Resolve a @pid: URI to directory path")
        .def("resolve_dir", [](rm::ResourceManager& mgr, const rm::URI& uri) {
            auto path = mgr.resolveDir(uri.to_string());
            if (path.empty()) {
                throw rm::RMError(rm::ErrorCode::NotFound, "Directory not found: " + uri.to_string());
            }
            return path.string();
        }, py::arg("uri"), "Resolve a @pid: URI to directory path")
        // __call__: URI-based access with metadata section support
        // Supports both old-style (pre, post) and new-style (pre, op1, op2) section names
        .def("__call__", [](rm::ResourceManager& mgr, const std::string& uri) -> py::object {
            static const std::set<std::string> METADATA_SECTIONS = {"pre", "post", "post2", "op1", "op2"};

            // Find last path component
            std::string clean_uri = uri;
            while (!clean_uri.empty() && clean_uri.back() == '/') {
                clean_uri.pop_back();
            }

            size_t last_slash = clean_uri.rfind('/');
            if (last_slash != std::string::npos) {
                std::string section = clean_uri.substr(last_slash + 1);

                if (METADATA_SECTIONS.count(section)) {
                    // Metadata access: fetch metadata.yaml and extract section
                    std::string base_uri = clean_uri.substr(0, last_slash);
                    std::string metadata_uri = base_uri + "/metadata.yaml";

                    auto handle = mgr.fetch(metadata_uri);
                    std::string content = std::string(handle.as_string());

                    // Parse YAML using Python's yaml module
                    py::module_ yaml = py::module_::import("yaml");
                    py::dict metadata = yaml.attr("safe_load")(content);

                    if (metadata.contains(section)) {
                        return metadata[py::str(section)];
                    }
                    throw std::runtime_error("Section '" + section + "' not found in metadata");
                }
            }

            // Default: return ResourceHandle
            return py::cast(mgr.fetch(uri));
        }, py::arg("uri"), "Access resource or metadata section via URI");
}
