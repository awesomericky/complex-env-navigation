//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "Environment.hpp"
#include "VectorizedEnvironment.hpp"

namespace py = pybind11;
using namespace raisim;

#ifndef ENVIRONMENT_NAME
  #define ENVIRONMENT_NAME RaisimGymEnv
#endif

PYBIND11_MODULE(RAISIMGYM_TORCH_ENV_NAME, m) {
  py::class_<VectorizedEnvironment<ENVIRONMENT>>(m, RSG_MAKE_STR(ENVIRONMENT_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<ENVIRONMENT>::init)
    .def("reset", &VectorizedEnvironment<ENVIRONMENT>::reset)
    .def("observe", &VectorizedEnvironment<ENVIRONMENT>::observe)
    .def("step", &VectorizedEnvironment<ENVIRONMENT>::step)
    .def("setSeed", &VectorizedEnvironment<ENVIRONMENT>::setSeed)
    .def("rewardInfo", &VectorizedEnvironment<ENVIRONMENT>::getRewardInfo)
    .def("close", &VectorizedEnvironment<ENVIRONMENT>::close)
    .def("isTerminalState", &VectorizedEnvironment<ENVIRONMENT>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<ENVIRONMENT>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<ENVIRONMENT>::setControlTimeStep)
    .def("getObDim", &VectorizedEnvironment<ENVIRONMENT>::getObDim)
    .def("getActionDim", &VectorizedEnvironment<ENVIRONMENT>::getActionDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<ENVIRONMENT>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<ENVIRONMENT>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<ENVIRONMENT>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<ENVIRONMENT>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<ENVIRONMENT>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<ENVIRONMENT>::curriculumUpdate)
    .def("reward_logging", &VectorizedEnvironment<ENVIRONMENT>::reward_logging)
    .def("contact_logging", &VectorizedEnvironment<ENVIRONMENT>::contact_logging)
    .def("torque_and_velocity_logging", &VectorizedEnvironment<ENVIRONMENT>::torque_and_velocity_logging)
    .def("set_user_command", &VectorizedEnvironment<ENVIRONMENT>::set_user_command)
    .def("initialize_n_step", &VectorizedEnvironment<ENVIRONMENT>::initialize_n_step)
    .def("coordinate_observe", &VectorizedEnvironment<ENVIRONMENT>::coordinate_observe)
    .def("partial_step", &VectorizedEnvironment<ENVIRONMENT>::partial_step)
    .def("partial_reset", &VectorizedEnvironment<ENVIRONMENT>::partial_reset)
    .def("visualize_desired_command_traj", &VectorizedEnvironment<ENVIRONMENT>::visualize_desired_command_traj)
    .def("visualize_modified_command_traj", &VectorizedEnvironment<ENVIRONMENT>::visualize_modified_command_traj)
    .def("set_goal", &VectorizedEnvironment<ENVIRONMENT>::set_goal)
    .def("parallel_set_goal", &VectorizedEnvironment<ENVIRONMENT>::parallel_set_goal)
    .def("computed_heading_direction", &VectorizedEnvironment<ENVIRONMENT>::computed_heading_direction)
    .def("single_env_collision_check", &VectorizedEnvironment<ENVIRONMENT>::single_env_collision_check)
    .def("parallel_env_collision_check", &VectorizedEnvironment<ENVIRONMENT>::parallel_env_collision_check)
    .def("analytic_planner_collision_check", &VectorizedEnvironment<ENVIRONMENT>::analytic_planner_collision_check)
    .def("visualize_analytic_planner", &VectorizedEnvironment<ENVIRONMENT>::visualize_analytic_planner)
    .def("visualize_waypoints", &VectorizedEnvironment<ENVIRONMENT>::visualize_waypoints)
    .def("getMapSize", &VectorizedEnvironment<ENVIRONMENT>::getMapSize)
    .def("getMapBound", &VectorizedEnvironment<ENVIRONMENT>::getMapBound)
    .def("getDepthImage", &VectorizedEnvironment<ENVIRONMENT>::getDepthImage)
    .def("getDepthImageHSize", &VectorizedEnvironment<ENVIRONMENT>::getDepthImageHSize)
    .def("getDepthImageVSize", &VectorizedEnvironment<ENVIRONMENT>::getDepthImageVSize)

    .def(py::pickle(
        [](const VectorizedEnvironment<ENVIRONMENT> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<ENVIRONMENT> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
        }
    ));
}
