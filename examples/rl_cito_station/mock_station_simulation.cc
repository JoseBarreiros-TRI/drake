#include <limits>
#include <iostream>

#include <gflags/gflags.h>

#include "drake/common/eigen_types.h"
#include "drake/common/find_resource.h"
#include "drake/common/is_approx_equal_abstol.h"
#include "drake/examples/rl_cito_station/rl_cito_station.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/lcmt_iiwa_command.hpp"
#include "drake/lcmt_iiwa_status.hpp"
#include "drake/lcmt_point_cloud.hpp"
#include "drake/lcmt_schunk_wsg_command.hpp"
#include "drake/lcmt_schunk_wsg_status.hpp"
#include "drake/manipulation/kuka_iiwa/iiwa_command_receiver.h"
#include "drake/manipulation/kuka_iiwa/iiwa_status_sender.h"
#include "drake/manipulation/schunk_wsg/schunk_wsg_lcm.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/perception/point_cloud_to_lcm.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/primitives/matrix_gain.h"
#include "drake/systems/sensors/image_to_lcm_image_array_t.h"
#include "drake/systems/sensors/optitrack_sender.h"

namespace drake {
namespace examples {
namespace rl_cito_station {
namespace {

// Runs a simulation of the manipulation station plant as a stand-alone
// simulation which mocks the network inputs and outputs of the real robot
// station.  This is a useful test in the transition from a single-process
// simulation to operating on the real robot hardware.

using Eigen::VectorXd;

DEFINE_double(target_realtime_rate, 1.0,
              "Playback speed.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_double(duration, std::numeric_limits<double>::infinity(),
              "Simulation duration.");
DEFINE_string(setup, "cito_rl",
              "Manipulation station type to simulate. "
              "Can be {cito_rl}");
DEFINE_bool(publish_point_cloud, false,
            "Whether to publish point clouds to LCM.  Note that per issue "
            "https://github.com/RobotLocomotion/drake/issues/12125 the "
            "simulated point cloud data will have registration errors.");

int do_main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  systems::DiagramBuilder<double> builder;

  // Create the "manipulation station".
  auto station = builder.AddSystem<RlCitoStation>();
 if (FLAGS_setup == "cito_rl") {
    station->SetupCitoRlStation();
    station->AddManipulandFromFile(
        "drake/examples/rl_cito_station/models/061_foam_brick.sdf",
        math::RigidTransform<double>(math::RotationMatrix<double>::Identity(),
                                     Eigen::Vector3d(0.6, 0, 0.15)),
        "box");
  } else {
    throw std::domain_error(
        "Unrecognized station type. Options are "
        "{cito_rl}.");
  }
  // TODO(russt): Load sdf objects specified at the command line.  Requires
  // #9747.
  station->Finalize();

  geometry::DrakeVisualizerd::AddToBuilder(
      &builder, station->GetOutputPort("query_object"));

  auto lcm = builder.AddSystem<systems::lcm::LcmInterfaceSystem>();

  auto iiwa_command_subscriber = builder.AddSystem(
      systems::lcm::LcmSubscriberSystem::Make<drake::lcmt_iiwa_command>(
          "IIWA_COMMAND", lcm));
  auto iiwa_command =
      builder.AddSystem<manipulation::kuka_iiwa::IiwaCommandReceiver>();
  builder.Connect(iiwa_command_subscriber->get_output_port(),
                  iiwa_command->get_message_input_port());
  builder.Connect(station->GetOutputPort("iiwa_position_measured"),
                  iiwa_command->get_position_measured_input_port());

  // Pull the positions out of the state.
  builder.Connect(iiwa_command->get_commanded_position_output_port(),
                  station->GetInputPort("iiwa_position"));
  builder.Connect(iiwa_command->get_commanded_torque_output_port(),
                  station->GetInputPort("iiwa_feedforward_torque"));

  auto iiwa_status =
      builder.AddSystem<manipulation::kuka_iiwa::IiwaStatusSender>();
  builder.Connect(station->GetOutputPort("iiwa_position_commanded"),
                  iiwa_status->get_position_commanded_input_port());
  builder.Connect(station->GetOutputPort("iiwa_position_measured"),
                  iiwa_status->get_position_measured_input_port());
  builder.Connect(station->GetOutputPort("iiwa_velocity_estimated"),
                  iiwa_status->get_velocity_estimated_input_port());
  builder.Connect(station->GetOutputPort("iiwa_torque_commanded"),
                  iiwa_status->get_torque_commanded_input_port());
  builder.Connect(station->GetOutputPort("iiwa_torque_measured"),
                  iiwa_status->get_torque_measured_input_port());
  builder.Connect(station->GetOutputPort("iiwa_torque_external"),
                  iiwa_status->get_torque_external_input_port());
  auto iiwa_status_publisher = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<drake::lcmt_iiwa_status>(
          "IIWA_STATUS", lcm, 0.005 /* publish period */));
  builder.Connect(iiwa_status->get_output_port(),
                  iiwa_status_publisher->get_input_port());

  if (FLAGS_setup == "cito_rl"){
    // mock Mocap
    auto& mbp_=station->get_multibody_plant();
    std::cout<<"instance index: "<<mbp_.GetModelInstanceByName("box") <<"body index: "<<mbp_.GetBodyByName("base_link",mbp_.GetModelInstanceByName("box")).index()<<std::endl;
    
    geometry::FrameId frame_id=mbp_.GetBodyFrameIdOrThrow(mbp_.GetBodyByName("base_link",mbp_.GetModelInstanceByName("box")).index());
    std::map<geometry::FrameId, std::pair<std::string, int>> frame_map;
    int optitrack_id = 0;
    frame_map[frame_id]=std::pair<std::string,int>("body_1",optitrack_id);
    
    const double fps_mocap = 100.0;
    auto optitrack_mock=
        builder.AddSystem<systems::sensors::OptitrackLcmFrameSender>(frame_map);   
    builder.Connect(station->GetOutputPort("geometry_poses"),
        optitrack_mock->get_input_port());

    auto optitrack_publisher = builder.AddSystem(
        systems::lcm::LcmPublisherSystem::Make<optitrack::optitrack_frame_t>(
            "OPTITRACK_FRAMES", lcm, 1.0 / fps_mocap));
    builder.Connect(optitrack_mock->get_lcm_output_port(),
                    optitrack_publisher->get_input_port());
  }

  auto diagram = builder.Build();

  systems::Simulator<double> simulator(*diagram);
  simulator.set_publish_every_time_step(false);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.AdvanceTo(FLAGS_duration);

  return 0;
}

}  // namespace
}  // namespace rl_cito_station
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  return drake::examples::rl_cito_station::do_main(argc, argv);
}
