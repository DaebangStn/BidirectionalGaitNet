/**
 * HDF DOF Remapper Tool
 *
 * Converts 56-DOF HDF motion files to 50-DOF (shoulderless) HDF motion files.
 *
 * DOF Index Mapping (56 → 50):
 *   - Pelvis→Head: indices 0-35 → copy directly to 0-35
 *   - ShoulderR: indices 36-38 → SKIP (3 DOF)
 *   - ArmR+ForeArmR+HandR: indices 39-45 → shift to 36-42
 *   - ShoulderL: indices 46-48 → SKIP (3 DOF)
 *   - ArmL+ForeArmL+HandL: indices 49-55 → shift to 43-49
 */

#include <H5Cpp.h>
#include <vector>
#include <string>
#include <iostream>
#include <getopt.h>
#include <filesystem>

namespace fs = std::filesystem;

constexpr int SRC_DOF = 56;
constexpr int DST_DOF = 50;

// Remap 56-DOF motion data to 50-DOF by removing shoulder joints
void remap56to50(const std::vector<float>& src, std::vector<float>& dst, int num_frames) {
    dst.resize(num_frames * DST_DOF);

    for (int f = 0; f < num_frames; f++) {
        int src_base = f * SRC_DOF;
        int dst_base = f * DST_DOF;

        // Copy indices 0-35 (Pelvis through Head) directly
        for (int i = 0; i < 36; i++) {
            dst[dst_base + i] = src[src_base + i];
        }

        // Skip indices 36-38 (ShoulderR)
        // Copy indices 39-45 (ArmR, ForeArmR, HandR) to 36-42
        for (int i = 0; i < 7; i++) {
            dst[dst_base + 36 + i] = src[src_base + 39 + i];
        }

        // Skip indices 46-48 (ShoulderL)
        // Copy indices 49-55 (ArmL, ForeArmL, HandL) to 43-49
        for (int i = 0; i < 7; i++) {
            dst[dst_base + 43 + i] = src[src_base + 49 + i];
        }
    }
}

// Copy a 1D float dataset from input to output
void copy1DDataset(H5::H5File& input_file, H5::H5File& output_file,
                   const std::string& dataset_name) {
    if (!H5Lexists(input_file.getId(), dataset_name.c_str(), H5P_DEFAULT)) {
        return;
    }

    H5::DataSet src_dataset = input_file.openDataSet(dataset_name);
    H5::DataSpace src_dataspace = src_dataset.getSpace();
    hsize_t dims[1];
    src_dataspace.getSimpleExtentDims(dims, nullptr);

    std::vector<float> data(dims[0]);
    src_dataset.read(data.data(), H5::PredType::NATIVE_FLOAT);

    H5::DataSpace dst_dataspace(1, dims);
    H5::DataSet dst_dataset = output_file.createDataSet(dataset_name, H5::PredType::NATIVE_FLOAT, dst_dataspace);
    dst_dataset.write(data.data(), H5::PredType::NATIVE_FLOAT);
}

// Copy variable-length string dataset
void copyVLStringDataset(H5::H5File& input_file, H5::H5File& output_file,
                         const std::string& dataset_name) {
    if (!H5Lexists(input_file.getId(), dataset_name.c_str(), H5P_DEFAULT)) {
        return;
    }

    H5::DataSet src_dataset = input_file.openDataSet(dataset_name);
    H5::DataSpace src_dataspace = src_dataset.getSpace();
    H5::DataType src_datatype = src_dataset.getDataType();

    hsize_t dims[1];
    src_dataspace.getSimpleExtentDims(dims, nullptr);

    std::vector<char*> data(dims[0]);
    src_dataset.read(data.data(), src_datatype);

    H5::DataSpace dst_dataspace(1, dims);
    H5::DataSet dst_dataset = output_file.createDataSet(dataset_name, src_datatype, dst_dataspace);
    dst_dataset.write(data.data(), src_datatype);

    H5Dvlen_reclaim(src_datatype.getId(), src_dataspace.getId(), H5P_DEFAULT, data.data());
}

bool processHDF(const std::string& input_path, const std::string& output_path, bool verbose) {
    try {
        H5::H5File input_file(input_path, H5F_ACC_RDONLY);
        H5::H5File output_file(output_path, H5F_ACC_TRUNC);

        // Read and remap /motions dataset
        {
            std::string dataset_name = "/motions";
            if (!H5Lexists(input_file.getId(), dataset_name.c_str(), H5P_DEFAULT)) {
                std::cerr << "Error: /motions dataset not found in input file" << std::endl;
                return false;
            }

            H5::DataSet src_dataset = input_file.openDataSet(dataset_name);
            H5::DataSpace src_dataspace = src_dataset.getSpace();
            hsize_t dims[2];
            src_dataspace.getSimpleExtentDims(dims, nullptr);

            int num_frames = static_cast<int>(dims[0]);
            int src_dof = static_cast<int>(dims[1]);

            if (src_dof != SRC_DOF) {
                std::cerr << "Error: Expected " << SRC_DOF << " DOF, got " << src_dof << std::endl;
                return false;
            }

            if (verbose) {
                std::cout << "Processing " << num_frames << " frames..." << std::endl;
                std::cout << "  Input DOF: " << src_dof << std::endl;
                std::cout << "  Output DOF: " << DST_DOF << std::endl;
            }

            // Read source data
            std::vector<float> src_data(num_frames * src_dof);
            src_dataset.read(src_data.data(), H5::PredType::NATIVE_FLOAT);

            // Remap to destination
            std::vector<float> dst_data;
            remap56to50(src_data, dst_data, num_frames);

            // Write output
            hsize_t dst_dims[2] = {static_cast<hsize_t>(num_frames), DST_DOF};
            H5::DataSpace dst_dataspace(2, dst_dims);
            H5::DataSet dst_dataset = output_file.createDataSet("/motions", H5::PredType::NATIVE_FLOAT, dst_dataspace);
            dst_dataset.write(dst_data.data(), H5::PredType::NATIVE_FLOAT);
        }

        // Also remap /param_motions if it exists (same format)
        {
            std::string dataset_name = "/param_motions";
            if (H5Lexists(input_file.getId(), dataset_name.c_str(), H5P_DEFAULT)) {
                H5::DataSet src_dataset = input_file.openDataSet(dataset_name);
                H5::DataSpace src_dataspace = src_dataset.getSpace();
                hsize_t dims[2];
                src_dataspace.getSimpleExtentDims(dims, nullptr);

                int num_frames = static_cast<int>(dims[0]);
                int src_dof = static_cast<int>(dims[1]);

                if (src_dof == SRC_DOF) {
                    if (verbose) {
                        std::cout << "Also remapping /param_motions (" << num_frames << " frames)..." << std::endl;
                    }

                    std::vector<float> src_data(num_frames * src_dof);
                    src_dataset.read(src_data.data(), H5::PredType::NATIVE_FLOAT);

                    std::vector<float> dst_data;
                    remap56to50(src_data, dst_data, num_frames);

                    hsize_t dst_dims[2] = {static_cast<hsize_t>(num_frames), DST_DOF};
                    H5::DataSpace dst_dataspace(2, dst_dims);
                    H5::DataSet dst_dataset = output_file.createDataSet("/param_motions", H5::PredType::NATIVE_FLOAT, dst_dataspace);
                    dst_dataset.write(dst_data.data(), H5::PredType::NATIVE_FLOAT);
                }
            }
        }

        // Copy other datasets unchanged
        copy1DDataset(input_file, output_file, "/phase");
        copy1DDataset(input_file, output_file, "/time");
        copy1DDataset(input_file, output_file, "/param_state");
        copyVLStringDataset(input_file, output_file, "/parameter_names");

        // Copy metadata if exists
        {
            std::string dataset_name = "/metadata";
            if (H5Lexists(input_file.getId(), dataset_name.c_str(), H5P_DEFAULT)) {
                H5::DataSet src_dataset = input_file.openDataSet(dataset_name);
                H5::DataType src_datatype = src_dataset.getDataType();
                H5::DataSpace src_dataspace = src_dataset.getSpace();

                std::string metadata;
                metadata.resize(src_datatype.getSize());
                src_dataset.read(&metadata[0], src_datatype);

                H5::StrType str_type(H5::PredType::C_S1, metadata.size() + 1);
                H5::DataSpace scalar_space(H5S_SCALAR);
                H5::DataSet dst_dataset = output_file.createDataSet("/metadata", str_type, scalar_space);
                dst_dataset.write(metadata.c_str(), str_type);
            }
        }

        input_file.close();
        output_file.close();

        return true;

    } catch (const H5::Exception& e) {
        std::cerr << "HDF5 Error: " << e.getDetailMsg() << std::endl;
        return false;
    }
}

void print_usage(const char* prog_name) {
    std::cout << "HDF DOF Remapper - Convert 56-DOF to 50-DOF motion files\n\n";
    std::cout << "Usage: " << prog_name << " -i INPUT -o OUTPUT [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -i, --input INPUT     Input HDF5 file (56 DOF)\n";
    std::cout << "  -o, --output OUTPUT   Output HDF5 file (50 DOF)\n";
    std::cout << "  -v, --verbose         Enable verbose output\n";
    std::cout << "  -h, --help            Show this help message\n\n";
    std::cout << "DOF Mapping:\n";
    std::cout << "  - Indices 0-35 (Pelvis→Head): copied directly\n";
    std::cout << "  - Indices 36-38 (ShoulderR): SKIPPED\n";
    std::cout << "  - Indices 39-45 (ArmR side): shifted to 36-42\n";
    std::cout << "  - Indices 46-48 (ShoulderL): SKIPPED\n";
    std::cout << "  - Indices 49-55 (ArmL side): shifted to 43-49\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog_name << " -i data/motion/walk.h5 -o data/motion/walk_50dof.h5\n";
    std::cout << "  " << prog_name << " -i input.h5 -o output.h5 -v\n";
}

int main(int argc, char** argv) {
    std::string input_file;
    std::string output_file;
    bool verbose = false;

    static struct option long_options[] = {
        {"input",   required_argument, 0, 'i'},
        {"output",  required_argument, 0, 'o'},
        {"verbose", no_argument,       0, 'v'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "i:o:vh", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i':
                input_file = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'v':
                verbose = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    // Validate required arguments
    if (input_file.empty() || output_file.empty()) {
        std::cerr << "Error: Both input (-i) and output (-o) files are required\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // Check input file exists
    if (!fs::exists(input_file)) {
        std::cerr << "Error: Input file not found: " << input_file << std::endl;
        return 1;
    }

    // Create output directory if needed
    fs::path output_path(output_file);
    if (output_path.has_parent_path()) {
        fs::create_directories(output_path.parent_path());
    }

    if (verbose) {
        std::cout << "Input:  " << input_file << std::endl;
        std::cout << "Output: " << output_file << std::endl;
    }

    bool success = processHDF(input_file, output_file, verbose);

    if (success) {
        std::cout << "Successfully converted " << input_file << " → " << output_file << std::endl;
        return 0;
    } else {
        std::cerr << "Conversion failed" << std::endl;
        return 1;
    }
}
