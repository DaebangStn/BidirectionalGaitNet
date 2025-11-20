#include "Environment.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>

namespace py = pybind11;

py::array_t<float> toNumPyArray(const Eigen::VectorXf &vec)
{
    int n = vec.rows();

    // Create array with explicit ownership - pybind11 manages memory
    auto array = py::array_t<float>(n);
    auto buf = array.request(true);
    auto ptr = static_cast<float *>(buf.ptr);

    // Use memcpy for better performance and safety
    std::memcpy(ptr, vec.data(), n * sizeof(float));

    return array;
}

py::array_t<float> toNumPyArray(const Eigen::VectorXd &vec)
{
    int n = vec.rows();

    // Create array with explicit ownership - pybind11 manages memory
    auto array = py::array_t<float>(n);
    auto buf = array.request(true);
    auto ptr = static_cast<float *>(buf.ptr);

    // Convert double to float element by element
    for (int i = 0; i < n; i++)
        ptr[i] = static_cast<float>(vec(i));

    return array;
}

py::array_t<float> toNumPyArray(const Eigen::MatrixXd &matrix)
{
    int n = matrix.rows();
    int m = matrix.cols();

    // Create array with explicit ownership - pybind11 manages memory
    auto array = py::array_t<float>({n, m});
    auto buf = array.request(true);
    auto ptr = static_cast<float *>(buf.ptr);

    // Convert double to float element by element (row-major order)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            ptr[i * m + j] = static_cast<float>(matrix(i, j));
        }
    }

    return array;
}

Eigen::VectorXd toEigenVector(const py::array_t<float> &array)
{
    Eigen::VectorXd vec(array.shape(0));

    py::buffer_info buf = array.request();
    float *srcs = reinterpret_cast<float *>(buf.ptr);

    for (int i = 0; i < array.shape(0); i++)
        vec(i) = (double)srcs[i];

    return vec;
}

class GymEnvManager : public Environment
{
public:
    GymEnvManager(std::string metadata) : Environment()
    {
        Environment::initialize(metadata);
        initializeBuffers();
    }

private:
    // Initialize muscle tuple buffer based on controller configuration
    void initializeBuffers()
    {
        if (Environment::isTwoLevelController())
        {
            int buffer_size = Environment::getUseCascading() ? 5 : 3;
            muscle_tuple_buffer.resize(buffer_size);
        }
    }

public:

    // Standard Gymnasium interface: reset returns (obs, info)
    py::tuple reset(double phase = -1.0)
    {
        Environment::reset(phase);
        py::array_t<float> obs = toNumPyArray(Environment::getState());
        py::dict info;  // Empty info dict for now
        return py::make_tuple(obs, info);
    }

    // Standard Gymnasium interface: step returns (obs, reward, terminated, truncated, info)
    py::tuple step(py::array_t<float> action)
    {
        Environment::setAction(toEigenVector(action));
        Environment::step();

        // Collect muscle tuples during step (if two-level controller)
        // Convert to numpy arrays immediately and store in buffer
        if (Environment::isTwoLevelController())
        {
            MuscleTuple mt = Environment::getRandomMuscleTuple();
            Eigen::VectorXd dt = Environment::getRandomDesiredTorque();

            // Check if muscle tuple data is valid (non-zero size)
            if (dt.size() > 0 && mt.JtA_reduced.rows() > 0 && mt.JtA.rows() > 0)
            {
                // Convert to numpy and store
                muscle_tuple_buffer[0].append(toNumPyArray(dt));
                muscle_tuple_buffer[1].append(toNumPyArray(mt.JtA_reduced));
                muscle_tuple_buffer[2].append(toNumPyArray(mt.JtA));

                if (Environment::getUseCascading())
                {
                    muscle_tuple_buffer[3].append(toNumPyArray(Environment::getRandomPrevOut()));
                    muscle_tuple_buffer[4].append(toNumPyArray(Environment::getRandomWeight()));
                }
            }
        }

        py::array_t<float> obs = toNumPyArray(Environment::getState());
        float reward = Environment::getReward();
        bool terminated = Environment::isTerminated();
        bool truncated = Environment::isTruncated();
        py::dict info = getInfoMap();

        // AUTO-RESET: If episode ended, reset and return fresh observation
        // This is required for proper episode tracking and AsyncVectorEnv compatibility
        if (terminated || truncated) {
            // Optional: Enable for debugging termination detection
            // std::cout << "Environment terminated or truncated" << std::endl;

            // Store final observation in info for AsyncVectorEnv compatibility
            info["final_observation"] = obs;

            // Reset environment to start new episode
            Environment::reset();

            // Return fresh observation from reset (not the terminal state)
            obs = toNumPyArray(Environment::getState());
        }

        return py::make_tuple(obs, reward, terminated, truncated, info);
    }

    py::list get_muscle_tuples()
    {
        // Return the buffer lists directly (already contains numpy arrays)
        py::list result;
        for (const auto& component_list : muscle_tuple_buffer)
        {
            result.append(component_list);
        }

        // Clear buffers
        int buffer_size = Environment::getUseCascading() ? 5 : 3;
        muscle_tuple_buffer.clear();
        muscle_tuple_buffer.resize(buffer_size);

        return result;
    }

    void update_muscle_weights(py::dict weights)
    {
        // Convert Python dict to torch-like format expected by C++
        Environment::setMuscleNetworkWeight(weights);
    }

    py::dict getInfoMap()
    {
        const auto& infoMap = Environment::getInfoMap();
        py::dict py_map;
        for (const auto& pair : infoMap)
        {
            py_map[py::cast(pair.first)] = pair.second;
        }
        return py_map;
    }

    py::array_t<float> getState() { return toNumPyArray(Environment::getState()); }
    py::array_t<float> getAction() { return toNumPyArray(Environment::getAction()); }

    int getNumMuscles() { return Environment::getCharacter()->getMuscles().size(); }
    int getNumMuscleDof() { return Environment::getCharacter()->getNumMuscleRelatedDof(); }

    py::array_t<float> getParamState() { return toNumPyArray(Environment::getParamState()); }
    py::array_t<float> getNormalizedParamState() { return toNumPyArray(Environment::getNormalizedParamState(Environment::getParamMin(), Environment::getParamMax())); }
    py::array_t<float> getPositions() { return toNumPyArray(Environment::getCharacter()->getSkeleton()->getPositions()); }
    py::array_t<float> posToSixDof(py::array_t<float> pos) { return toNumPyArray(Environment::getCharacter()->posToSixDof(toEigenVector(pos))); }
    py::array_t<float> sixDofToPos(py::array_t<float> raw_pos) { return toNumPyArray(Environment::getCharacter()->sixDofToPos(toEigenVector(raw_pos))); }

    py::array_t<float> getMirrorParamState(py::array_t<float> param)
    {
        Eigen::VectorXd cur_paramstate = Environment::getParamState();
        Environment::setNormalizedParamState(toEigenVector(param), false, true);
        Eigen::VectorXd res = Environment::getNormalizedParamState(Environment::getParamMin(), Environment::getParamMax(), true);
        Environment::setParamState(cur_paramstate, false, true);
        return toNumPyArray(res);
    }

    py::array_t<float> getMirrorPositions(py::array_t<float> pos) { return toNumPyArray(getCharacter()->getMirrorPosition(toEigenVector(pos))); }

    py::array_t<float> getParamStateFromNormalized(py::array_t<float> normalized_param)
    {
        return toNumPyArray(Environment::getParamStateFromNormalized(toEigenVector(normalized_param)));
    }

    py::array_t<float> getNormalizedParamStateFromParam(py::array_t<float> param)
    {
        return toNumPyArray(Environment::getNormalizedParamStateFromParam(toEigenVector(param)));
    }

    py::array_t<float> getNormalizedParamSample()
    {
        return toNumPyArray(Environment::getNormalizedParamStateFromParam(Environment::getParamSample()));
    }

private:
    // Muscle tuple buffer: stores Python lists of numpy arrays
    // [component_idx] -> py::list of numpy arrays
    std::vector<py::list> muscle_tuple_buffer;
};

PYBIND11_MODULE(gymenv, m)
{
    py::class_<GymEnvManager>(m, "GymEnvManager")
        .def(py::init<std::string>())

        // Standard Gymnasium interface
        .def("reset", &GymEnvManager::reset, py::arg("phase") = -1.0)
        .def("step", &GymEnvManager::step)
        .def("getReward", &GymEnvManager::getReward)
        .def("getInfoMap", &GymEnvManager::getInfoMap)
        .def("getState", &GymEnvManager::getState)
        .def("getAction", &GymEnvManager::getAction)

        // Option C hierarchical control interface
        .def("get_muscle_tuples", &GymEnvManager::get_muscle_tuples)
        .def("update_muscle_weights", &GymEnvManager::update_muscle_weights)

        // Configuration getters
        .def("getNumAction", &GymEnvManager::getNumAction)
        .def("getNumActuatorAction", &GymEnvManager::getNumActuatorAction)
        .def("getNumMuscles", &GymEnvManager::getNumMuscles)
        .def("getNumMuscleDof", &GymEnvManager::getNumMuscleDof)
        .def("getMetadata", &GymEnvManager::getMetadata)
        .def("isTwoLevelController", &GymEnvManager::isTwoLevelController)
        .def("getLearningStd", &GymEnvManager::getLearningStd)
        .def("getUseCascading", &GymEnvManager::getUseCascading)

        // Parameter management
        .def("getParamState", &GymEnvManager::getParamState)
        .def("updateParamState", &GymEnvManager::updateParamState)
        .def("getNormalizedParamState", &GymEnvManager::getNormalizedParamState)
        .def("getNormalizedPhase", &GymEnvManager::getNormalizedPhase)
        .def("getWorldPhase", &GymEnvManager::getWorldPhase)
        .def("getPositions", &GymEnvManager::getPositions)
        .def("posToSixDof", &GymEnvManager::posToSixDof)
        .def("sixDofToPos", &GymEnvManager::sixDofToPos)
        .def("getMirrorParamState", &GymEnvManager::getMirrorParamState)
        .def("getMirrorPositions", &GymEnvManager::getMirrorPositions)
        .def("getParamStateFromNormalized", &GymEnvManager::getParamStateFromNormalized)
        .def("getNormalizedParamStateFromParam", &GymEnvManager::getNormalizedParamStateFromParam)
        .def("getNumKnownParam", &GymEnvManager::getNumKnownParam)
        .def("getNormalizedParamSample", &GymEnvManager::getNormalizedParamSample)
        .def("getParamMin", &GymEnvManager::getParamMin)
        .def("getParamMax", &GymEnvManager::getParamMax)
        ;
}
