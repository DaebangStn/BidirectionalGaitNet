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

class EnvManager : public Environment
{
public:
    EnvManager(std::string filepath) : Environment(filepath) {}
    py::array_t<float> getState() { return toNumPyArray(Environment::getState()); }
    py::array_t<float> getAction() { return toNumPyArray(Environment::getAction()); }
    py::list getRandomMuscleTuple()
    {
        MuscleTuple mt = Environment::getRandomMuscleTuple();
        Eigen::VectorXd dt = Environment::getRandomDesiredTorque();

        py::list py_mt;
        py_mt.append(toNumPyArray(dt));
        py_mt.append(toNumPyArray(mt.JtA_reduced));
        py_mt.append(toNumPyArray(mt.JtA));
        if (Environment::getUseCascading())
        {
            py_mt.append(toNumPyArray(Environment::getRandomPrevOut()));
            py_mt.append(toNumPyArray(Environment::getRandomWeight()));
        }
        return py_mt;
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
    void setAction(py::array_t<float> action) { Environment::setAction(toEigenVector(action)); }
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
        Environment::setNormalizedParamState(toEigenVector(param));
        Eigen::VectorXd res = Environment::getNormalizedParamState(Environment::getParamMin(), Environment::getParamMax(), true);
        Environment::setParamState(cur_paramstate);
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

    // getParamSample
    py::array_t<float> getNormalizedParamSample()
    {
        return toNumPyArray(Environment::getNormalizedParamStateFromParam(Environment::getParamSample()));
    }
};

PYBIND11_MODULE(pysim, m)
{
    py::class_<EnvManager>(m, "EnvManager")
        .def(py::init<std::string>())  // filepath to YAML config
        // .def("initialize", &EnvManager::initialize)
        .def("setAction", &EnvManager::setAction)
        .def("step", &EnvManager::step)
        .def("reset", &EnvManager::reset, py::arg("phase") = -1.0)
        .def("isTerminated", &EnvManager::isTerminated)
        .def("isTruncated", &EnvManager::isTruncated)
        .def("getReward", &EnvManager::getReward)
        .def("getInfoMap", &EnvManager::getInfoMap)
        .def("getState", &EnvManager::getState)
        .def("getAction", &EnvManager::getAction)

        .def("getNumAction", &EnvManager::getNumAction)
        .def("getNumActuatorAction", &EnvManager::getNumActuatorAction)
        .def("getNumMuscles", &EnvManager::getNumMuscles)
        .def("getNumMuscleDof", &EnvManager::getNumMuscleDof)

        .def("getMetadata", &EnvManager::getMetadata)
        .def("getRandomMuscleTuple", &EnvManager::getRandomMuscleTuple)
        .def("getUseMuscle", &EnvManager::getUseMuscle)
        .def("setMuscleNetwork", &EnvManager::setMuscleNetwork)
        .def("setMuscleNetworkWeight", &EnvManager::setMuscleNetworkWeight)
        .def("isTwoLevelController", &EnvManager::isTwoLevelController)

        .def("getUseCascading", &EnvManager::getUseCascading)

        .def("getParamState", &EnvManager::getParamState)
        .def("updateParamState", &EnvManager::updateParamState)
         
        // For Rollout (Forward GaitNet)
        .def("getNormalizedParamState", &EnvManager::getNormalizedParamState)

        .def("getPositions", &EnvManager::getPositions)

        .def("posToSixDof", &EnvManager::posToSixDof)
        .def("sixDofToPos", &EnvManager::sixDofToPos)

        .def("getMirrorParamState", &EnvManager::getMirrorParamState)
        .def("getMirrorPositions", &EnvManager::getMirrorPositions)
        .def("getParamStateFromNormalized", &EnvManager::getParamStateFromNormalized)
        .def("getNormalizedParamStateFromParam", &EnvManager::getNormalizedParamStateFromParam)
        .def("getNumKnownParam", &EnvManager::getNumKnownParam)

        .def("getNormalizedParamSample", &EnvManager::getNormalizedParamSample)

        // get min v
        .def("getParamMin", &EnvManager::getParamMin)
        .def("getParamMax", &EnvManager::getParamMax)

        ;
}
