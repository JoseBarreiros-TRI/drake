#pragma once

#include <memory>
#include <string>
#include <vector>

#include "drake/lcm/drake_lcm.h"
#include "drake/lcm/drake_lcm_interface.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"

namespace drake {
namespace examples {
namespace rl_cito_station {

/// A System that connects via message-passing to the hardware manipulation
/// station.
///
/// Note: Users must call Connect() after initialization.
///
/// @{
///
/// @system
/// name: RlCitoStationHardwareInterface
/// input_ports:
/// - iiwa_position
/// - iiwa_feedforward_torque
/// output_ports:
/// - iiwa_position_commanded
/// - iiwa_position_measured
/// - iiwa_velocity_estimated
/// - iiwa_torque_commanded
/// - iiwa_torque_measured
/// - iiwa_torque_external
/// @endsystem
///
/// @ingroup rl_cito_station_systems
/// @}
///
class RlCitoStationHardwareInterface : public systems::Diagram<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(RlCitoStationHardwareInterface)

  RlCitoStationHardwareInterface(
      bool has_optitrack=true,
      const Eigen::Matrix3d& A = Eigen::Matrix3d::Identity());

  /// Starts a thread to receive network messages, and blocks execution until
  /// the first messages have been received.
  void Connect(bool wait_for_optitrack=true );

  /// For parity with RlCitoStation, we maintain a MultibodyPlant of
  /// the IIWA arm, with the lumped-mass equivalent spatial inertia of the
  const multibody::MultibodyPlant<double>& get_controller_plant() const {
    return *owned_controller_plant_;
  }

  /// Gets the number of joints in the IIWA (only -- does not include the
  /// gripper).
  int num_iiwa_joints() const;

 private:
  std::unique_ptr<multibody::MultibodyPlant<double>> owned_controller_plant_;
  std::unique_ptr<lcm::DrakeLcm> owned_lcm_;
  systems::lcm::LcmSubscriberSystem* optitrack_subscriber;
  systems::lcm::LcmSubscriberSystem* iiwa_status_subscriber_;

  multibody::ModelInstanceIndex iiwa_model_instance_{};
};

}  // namespace rl_cito_station
}  // namespace examples
}  // namespace drake
