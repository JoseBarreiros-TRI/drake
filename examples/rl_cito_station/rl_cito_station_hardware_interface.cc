#include "drake/examples/rl_cito_station/rl_cito_station_hardware_interface.h"

#include <iostream>
#include <utility>
#include <limits>

#include "drake/common/find_resource.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/lcmt_iiwa_command.hpp"
#include "drake/lcmt_iiwa_status.hpp"
#include "drake/lcmt_image_array.hpp"
#include "drake/manipulation/kuka_iiwa/iiwa_command_sender.h"
#include "drake/manipulation/kuka_iiwa/iiwa_status_receiver.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/primitives/pass_through.h"
#include "drake/systems/sensors/lcm_image_array_to_images.h"
#include "drake/systems/sensors/optitrack_receiver.h"
#include "drake/systems/sensors/optitrack_sender.h"


namespace drake {
namespace examples {
namespace rl_cito_station {

using Eigen::Vector3d;
using Eigen::Matrix3d;
using multibody::MultibodyPlant;
using multibody::Parser;
using systems::Context;

template <typename T>

class ApplyTransformToPose final : public systems::LeafSystem<T> {
 public:
  explicit ApplyTransformToPose(){
    input_ = &this->DeclareAbstractInputPort(
        "input_pose", Value<math::RigidTransformd>());
    this->DeclareAbstractOutputPort(
        "output_pose", &ApplyTransformToPose<T>::Applicator);
    //TODO(jose-tri): change this as an argument. 
    A_ << 1,0,0,0,1,0,0,0,1; 
  }
  void Applicator(const Context<T>& context,
                 math::RigidTransform<T>* output) const {
    const math::RigidTransform<T>& pose =
        input_->Eval<math::RigidTransform<T>>(context);
    
    *output = pose;
    
    output->set_rotation(pose.rotation());
    output->set_translation(A_ * pose.translation());
  }

 private:
  Matrix3d A_;
  const systems::InputPort<double>* input_{};
 
};

// TODO(russt): Consider taking DrakeLcmInterface as an argument instead of
// (only) constructing one internally.
// TODO(jose-tri): Add argument for mock_hardware. 
RlCitoStationHardwareInterface::RlCitoStationHardwareInterface(
    bool has_optitrack)
    : owned_controller_plant_(std::make_unique<MultibodyPlant<double>>(0.0)),
      owned_lcm_(new lcm::DrakeLcm()){
  systems::DiagramBuilder<double> builder;

  auto lcm = builder.AddSystem<systems::lcm::LcmInterfaceSystem>(
      owned_lcm_.get());

  // Publish IIWA command.
  auto iiwa_command_sender =
      builder.AddSystem<manipulation::kuka_iiwa::IiwaCommandSender>();
  auto iiwa_command_publisher = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<drake::lcmt_iiwa_command>(
          "IIWA_COMMAND", lcm, 0.005
          /* publish period: IIWA driver won't respond faster than 200Hz */));
  builder.ExportInput(iiwa_command_sender->get_position_input_port(),
                      "iiwa_position");
  builder.ExportInput(iiwa_command_sender->get_torque_input_port(),
                      "iiwa_feedforward_torque");
  builder.Connect(iiwa_command_sender->get_output_port(),
                  iiwa_command_publisher->get_input_port());

  // Receive IIWA status and populate the output ports.
  auto iiwa_status_receiver =
      builder.AddSystem<manipulation::kuka_iiwa::IiwaStatusReceiver>();
  iiwa_status_subscriber_ = builder.AddSystem(
      systems::lcm::LcmSubscriberSystem::Make<drake::lcmt_iiwa_status>(
          "IIWA_STATUS", lcm));

  builder.ExportOutput(
      iiwa_status_receiver->get_position_commanded_output_port(),
      "iiwa_position_commanded");
  builder.ExportOutput(
      iiwa_status_receiver->get_position_measured_output_port(),
      "iiwa_position_measured");
  builder.ExportOutput(
      iiwa_status_receiver->get_velocity_estimated_output_port(),
      "iiwa_velocity_estimated");
  builder.ExportOutput(iiwa_status_receiver->get_torque_commanded_output_port(),
                       "iiwa_torque_commanded");
  builder.ExportOutput(iiwa_status_receiver->get_torque_measured_output_port(),
                       "iiwa_torque_measured");
  builder.ExportOutput(iiwa_status_receiver->get_torque_external_output_port(),
                       "iiwa_torque_external");
  builder.Connect(iiwa_status_subscriber_->get_output_port(),
                  iiwa_status_receiver->get_input_port());
  
  if (has_optitrack==true){
    //subscribe to box pose through LCM
    optitrack_subscriber =
        builder.AddSystem(
            systems::lcm::LcmSubscriberSystem::Make<optitrack::optitrack_frame_t>(
                "OPTITRACK_FRAMES", lcm ));
    int optitrack_id = 0;            
    std::map<int,std::string> frame_map{{optitrack_id, "body_1"}};;

    auto optitrack_decoder =
        builder.AddSystem<systems::sensors::OptitrackReceiver>(frame_map);

    builder.Connect(optitrack_subscriber->get_output_port(),
                    optitrack_decoder->get_input_port());

    // apply pose transform
    auto pose_transform=builder.AddSystem<ApplyTransformToPose<double>>();
    builder.Connect(optitrack_decoder->GetOutputPort("body_1"),
          pose_transform->get_input_port());
    builder.ExportOutput(
        pose_transform->get_output_port(),
        "optitrack_manipuland_pose");

    // builder.ExportOutput(
    //     optitrack_decoder->GetOutputPort("body_1"),
    //     "optitrack_manipuland_pose");
  }
  builder.BuildInto(this);
  this->set_name("rl_cito_station_hardware_interface");

  // Build the controller's version of the plant, which only contains the
  // IIWA and the equivalent inertia of the gripper.
  const std::string iiwa_sdf_path = FindResourceOrThrow(
      "drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf");
  Parser parser(owned_controller_plant_.get());
  iiwa_model_instance_ = parser.AddModelFromFile(iiwa_sdf_path, "iiwa");

  // TODO(russt): Provide API for changing the base coordinates of the plant.
  owned_controller_plant_->WeldFrames(owned_controller_plant_->world_frame(),
                                      owned_controller_plant_->GetFrameByName(
                                          "iiwa_link_0", iiwa_model_instance_),
                                      math::RigidTransformd::Identity());
  owned_controller_plant_->Finalize();
}

void RlCitoStationHardwareInterface::Connect(bool wait_for_optitrack) {
  drake::lcm::DrakeLcmInterface* const lcm = owned_lcm_.get();
  auto wait_for_new_message = [lcm](const auto& lcm_sub) {
    std::cout << "Waiting for " << lcm_sub.get_channel_name()
              << " message..." << std::flush;
    const int orig_count = lcm_sub.GetInternalMessageCount();
    LcmHandleSubscriptionsUntil(lcm, [&]() {
        return lcm_sub.GetInternalMessageCount() > orig_count;
      }, 10 /* timeout_millis */);
    std::cout << "Received!" << std::endl;
  };

  wait_for_new_message(*iiwa_status_subscriber_);
  if (wait_for_optitrack==true){
    wait_for_new_message(*optitrack_subscriber);
  }
}

int RlCitoStationHardwareInterface::num_iiwa_joints() const {
  DRAKE_DEMAND(iiwa_model_instance_.is_valid());
  return owned_controller_plant_->num_positions(iiwa_model_instance_);
}

}  // namespace rl_cito_station
}  // namespace examples
}  // namespace drake
