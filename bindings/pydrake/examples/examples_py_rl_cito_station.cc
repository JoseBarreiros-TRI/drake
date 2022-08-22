#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "drake/bindings/pydrake/common/deprecation_pybind.h"
#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/examples/examples_py.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/examples/rl_cito_station/rl_cito_station.h"
#include "drake/examples/rl_cito_station/rl_cito_station_hardware_interface.h"  // noqa

using std::make_unique;
using std::unique_ptr;
using std::vector;

namespace drake {
namespace pydrake {
namespace internal {

void DefineExamplesRlCitoStation(py::module m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::systems;
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::examples::rl_cito_station;
  constexpr auto& doc = pydrake_doc.drake.examples.rl_cito_station;

  // RlCitoStation currently only supports double.
  using T = double;

  py::enum_<IiwaCollisionModel_>(
      m, "IiwaCollisionModel_", doc.IiwaCollisionModel_.doc)
      .value("kNoCollision", IiwaCollisionModel_::kNoCollision,
          doc.IiwaCollisionModel_.kNoCollision.doc)
      .value("kBoxCollision", IiwaCollisionModel_::kBoxCollision,
          doc.IiwaCollisionModel_.kBoxCollision.doc)
      .export_values();

  {
    using Class = RlCitoStation<T>;
    const auto& cls_doc = doc.RlCitoStation;

    py::class_<Class, Diagram<T>> cls(m, "RlCitoStation", cls_doc.doc);
    cls  // BR
        .def(py::init<double, std::string, std::string>(), py::arg("time_step") = 0.002, py::arg("contact_model")="point", py::arg("contact_solver")="tamsi", cls_doc.ctor.doc)
        .def("SetupCitoRlStation",
            &Class::SetupCitoRlStation,
            py::arg("collision_model") = IiwaCollisionModel_::kBoxCollision,
            cls_doc.SetupCitoRlStation.doc)
        .def("AddManipulandFromFile", &Class::AddManipulandFromFile,
            py::arg("model_file"), py::arg("X_WObject"),py::arg("manipuland_name"),
            cls_doc.AddManipulandFromFile.doc)
        .def("RegisterIiwaControllerModel", &Class::RegisterIiwaControllerModel,
            cls_doc.RegisterIiwaControllerModel.doc)
        .def("Finalize", py::overload_cast<>(&Class::Finalize),
            cls_doc.Finalize.doc)
        .def("num_iiwa_joints", &Class::num_iiwa_joints,
            cls_doc.num_iiwa_joints.doc)
        .def("get_multibody_plant", &Class::get_multibody_plant,
            py_rvp::reference_internal, cls_doc.get_multibody_plant.doc)
        .def("get_mutable_multibody_plant", &Class::get_mutable_multibody_plant,
            py_rvp::reference_internal, cls_doc.get_mutable_multibody_plant.doc)
        .def("get_scene_graph", &Class::get_scene_graph,
            py_rvp::reference_internal, cls_doc.get_scene_graph.doc)
        .def("get_mutable_scene_graph", &Class::get_mutable_scene_graph,
            py_rvp::reference_internal, cls_doc.get_mutable_scene_graph.doc)
        .def("get_controller_plant", &Class::get_controller_plant,
            py_rvp::reference_internal, cls_doc.get_controller_plant.doc)
        .def("GetIiwaPosition", &Class::GetIiwaPosition,
            cls_doc.GetIiwaPosition.doc)
        .def("SetIiwaPosition",
            overload_cast_explicit<void, systems::Context<T>*,
                const Eigen::Ref<const VectorX<T>>&>(&Class::SetIiwaPosition),
            py::arg("station_context"), py::arg("q"),
            cls_doc.SetIiwaPosition.doc_2args)
        .def("GetIiwaVelocity", &Class::GetIiwaVelocity,
            cls_doc.GetIiwaVelocity.doc)
        .def("SetIiwaVelocity",
            overload_cast_explicit<void, systems::Context<T>*,
                const Eigen::Ref<const VectorX<T>>&>(&Class::SetIiwaVelocity),
            py::arg("station_context"), py::arg("v"),
            cls_doc.SetIiwaVelocity.doc_2args)
        .def("SetIiwaPositionGains", &Class::SetIiwaPositionGains,
            cls_doc.SetIiwaPositionGains.doc)
        .def("SetIiwaVelocityGains", &Class::SetIiwaVelocityGains,
            cls_doc.SetIiwaVelocityGains.doc)
        .def("SetIiwaIntegralGains", &Class::SetIiwaIntegralGains,
            cls_doc.SetIiwaIntegralGains.doc);
  }

  py::class_<RlCitoStationHardwareInterface, Diagram<double>>(m,
      "RlCitoStationHardwareInterface",
      doc.RlCitoStationHardwareInterface.doc)
      .def(py::init<bool,  const Eigen::Matrix3d&>(),
          py::arg("has_optitrack") = true,
          py::arg("optitrack_frame_transform") = Eigen::Matrix3d::Identity(),
          doc.RlCitoStationHardwareInterface.ctor.doc)
      .def("Connect", &RlCitoStationHardwareInterface::Connect,
          py::arg("wait_for_optitrack") = true,
          doc.RlCitoStationHardwareInterface.Connect.doc)
      .def("get_controller_plant",
          &RlCitoStationHardwareInterface::get_controller_plant,
          py_rvp::reference_internal,
          doc.RlCitoStationHardwareInterface.get_controller_plant.doc)
      .def("num_iiwa_joints",
          &RlCitoStationHardwareInterface::num_iiwa_joints,
          doc.RlCitoStationHardwareInterface.num_iiwa_joints.doc);
}

}  // namespace internal
}  // namespace pydrake
}  // namespace drake
