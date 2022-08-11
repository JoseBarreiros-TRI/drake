#include "drake/examples/rl_cito_station/rl_cito_station.h"

#include <map>

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_no_throw.h"
#include "drake/geometry/test_utilities/dummy_render_engine.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/primitives/discrete_derivative.h"
#include "drake/systems/sensors/image.h"

namespace drake {
namespace examples {
namespace rl_cito_station {
namespace {

using Eigen::Vector2d;
using Eigen::VectorXd;
using geometry::internal::DummyRenderEngine;
using multibody::RevoluteJoint;
using systems::BasicVector;

GTEST_TEST(RlCitoStationTest, CheckPlantBasics) {
  RlCitoStation<double> station(0.001);
  station.SetupCitoRlStation();
  multibody::Parser parser(&station.get_mutable_multibody_plant(),
                           &station.get_mutable_scene_graph());
  parser.AddModelFromFile(
      FindResourceOrThrow("drake/examples/rl_cito_station/models"
                          "/061_foam_brick.sdf"),
      "object");
  station.Finalize();

  auto& plant = station.get_multibody_plant();
  EXPECT_EQ(plant.num_actuated_dofs(), 7);  // 7 iiwa + 2 wsg.

  auto context = station.CreateDefaultContext();
  auto& plant_context = station.GetSubsystemContext(plant, *context);
  VectorXd q = VectorXd::LinSpaced(7, 0.1, 0.7),
           v = VectorXd::LinSpaced(7, 1.1, 1.7),
           q_command = VectorXd::LinSpaced(7, 2.1, 2.7),
           tau_ff = VectorXd::LinSpaced(7, 3.1, 3.7);

  // Set positions and read them back out, multiple ways.
  station.SetIiwaPosition(context.get(), q);
  EXPECT_TRUE(CompareMatrices(q, station.GetIiwaPosition(*context)));
  EXPECT_TRUE(CompareMatrices(q, station.GetOutputPort("iiwa_position_measured")
                                     .Eval<BasicVector<double>>(*context)
                                     .get_value()));
  for (int i = 0; i < 7; i++) {
    EXPECT_EQ(q(i), plant
                        .template GetJointByName<RevoluteJoint>(
                            "iiwa_joint_" + std::to_string(i + 1))
                        .get_angle(plant_context));
  }

  // Set velocities and read them back out, multiple ways.
  station.SetIiwaVelocity(context.get(), v);
  EXPECT_TRUE(CompareMatrices(v, station.GetIiwaVelocity(*context)));
  EXPECT_TRUE(
      CompareMatrices(v, station.GetOutputPort("iiwa_velocity_estimated")
                             .Eval<BasicVector<double>>(*context)
                             .get_value()));
  for (int i = 0; i < 7; i++) {
    EXPECT_EQ(v(i), plant
                        .template GetJointByName<RevoluteJoint>(
                            "iiwa_joint_" + std::to_string(i + 1))
                        .get_angular_rate(plant_context));
  }

  // Check position command pass through.
  station.GetInputPort("iiwa_position").FixValue(context.get(), q_command);
  EXPECT_TRUE(CompareMatrices(q_command,
                              station.GetOutputPort("iiwa_position_commanded")
                                  .Eval<BasicVector<double>>(*context)
                                  .get_value()));

  // Check feedforward_torque command.
  VectorXd tau_with_no_ff = station.GetOutputPort("iiwa_torque_commanded")
                                .Eval<BasicVector<double>>(*context)
                                .get_value();
  // Confirm that default values are zero.
  station.GetInputPort("iiwa_feedforward_torque")
      .FixValue(context.get(), VectorXd::Zero(7));
  EXPECT_TRUE(CompareMatrices(tau_with_no_ff,
                              station.GetOutputPort("iiwa_torque_commanded")
                                  .Eval<BasicVector<double>>(*context)
                                  .get_value()));
  station.GetInputPort("iiwa_feedforward_torque")
      .FixValue(context.get(), tau_ff);
  EXPECT_TRUE(CompareMatrices(tau_with_no_ff + tau_ff,
                              station.GetOutputPort("iiwa_torque_commanded")
                                  .Eval<BasicVector<double>>(*context)
                                  .get_value()));

  // Check iiwa_torque_commanded == iiwa_torque_measured.
  EXPECT_TRUE(CompareMatrices(station.GetOutputPort("iiwa_torque_commanded")
                                  .Eval<BasicVector<double>>(*context)
                                  .get_value(),
                              station.GetOutputPort("iiwa_torque_measured")
                                  .Eval<BasicVector<double>>(*context)
                                  .get_value()));

  // Check that iiwa_torque_external == 0 (no contact).
  EXPECT_TRUE(station.GetOutputPort("iiwa_torque_external")
                  .Eval<BasicVector<double>>(*context)
                  .get_value()
                  .isZero());

  // Check that the additional output ports exist and are spelled correctly.
  DRAKE_EXPECT_NO_THROW(station.GetOutputPort("contact_results"));
  DRAKE_EXPECT_NO_THROW(station.GetOutputPort("plant_continuous_state"));
}

// Partially check M(q)vdot ≈ Mₑ(q)vdot_desired + τ_feedforward + τ_external
// by setting the right side to zero and confirming that vdot ≈ 0.
GTEST_TEST(RlCitoStationTest, CheckDynamics) {
  const double kTimeStep = 0.002;
  RlCitoStation<double> station(kTimeStep);
  station.SetupCitoRlStation();
  station.Finalize();

  auto context = station.CreateDefaultContext();

  // Expect continuous state from the integral term in the PID from the
  // inverse dynamics controller.
  EXPECT_EQ(context->num_continuous_states(), 7);

  const auto& plant = station.get_multibody_plant();

  const VectorXd iiwa_position = VectorXd::LinSpaced(7, 0.735, 0.983);
  const VectorXd iiwa_velocity = VectorXd::LinSpaced(7, -1.23, 0.456);
  station.SetIiwaPosition(context.get(), iiwa_position);
  station.SetIiwaVelocity(context.get(), iiwa_velocity);

  station.GetInputPort("iiwa_position").FixValue(context.get(),
                        iiwa_position);
  station.GetInputPort("iiwa_feedforward_torque").FixValue(
      context.get(), VectorXd::Zero(7));

  // Set desired position to actual position and the desired velocity to the
  // actual velocity.
  const auto& position_to_state = dynamic_cast<
      const systems::StateInterpolatorWithDiscreteDerivative<double>&>(
      station.GetSubsystemByName("desired_state_from_position"));
  auto& position_to_state_context =
      station.GetMutableSubsystemContext(position_to_state, context.get());
  position_to_state.set_initial_state(&position_to_state_context, iiwa_position,
                                      iiwa_velocity);
  // Ensure that integral terms are zero.
  context->get_mutable_continuous_state_vector().SetZero();

  // Check that iiwa_torque_external == 0 (no contact).
  EXPECT_TRUE(station.GetOutputPort("iiwa_torque_external")
                  .Eval<BasicVector<double>>(*context)
                  .get_value()
                  .isZero());

  auto next_state = station.AllocateDiscreteVariables();
  station.CalcDiscreteVariableUpdates(*context, next_state.get());

  // Check that vdot ≈ 0 by checking that next velocity ≈ velocity.
  const auto& base_joint =
      plant.GetJointByName<multibody::RevoluteJoint>("iiwa_joint_1");
  const int iiwa_velocity_start =
      plant.num_positions() + base_joint.velocity_start();
  VectorXd next_velocity =
      station.GetSubsystemDiscreteValues(plant, *next_state)
          .value()
          .segment<7>(iiwa_velocity_start);

  // Note: This tolerance could be much smaller if the wsg was not attached.
  const double kTolerance = 1e-4;  // rad/sec.
  EXPECT_TRUE(CompareMatrices(iiwa_velocity, next_velocity, kTolerance));
}

GTEST_TEST(RlCitoStationTest, CheckCollisionVariants) {
  RlCitoStation<double> station1(0.002);
  station1.SetupCitoRlStation(IiwaCollisionModel_::kNoCollision);

  // In this variant, there are collision geometries from the world and the
  // gripper, but not from the iiwa.
  const int num_collisions =
      station1.get_multibody_plant().num_collision_geometries();

  RlCitoStation<double> station2(0.002);
  station2.SetupCitoRlStation(IiwaCollisionModel_::kBoxCollision);
  // Check for additional collision elements (one for each link, which includes
  // the base).
  EXPECT_EQ(station2.get_multibody_plant().num_collision_geometries(),
            num_collisions + 8);

  // The controlled model does not register with a scene graph, so has zero
  // collisions.
  EXPECT_EQ(station2.get_controller_plant().num_collision_geometries(), 0);
}

GTEST_TEST(RlCitoStationTest, AddManipulandFromFile) {
  RlCitoStation<double> station(0.002);
  const int num_base_instances =
      station.get_multibody_plant().num_model_instances();

  station.AddManipulandFromFile(
      "drake/manipulation/models/ycb/sdf/003_cracker_box.sdf",
      math::RigidTransform<double>::Identity());

  // Check that the cracker box was added.
  EXPECT_EQ(station.get_multibody_plant().num_model_instances(),
            num_base_instances + 1);

  station.AddManipulandFromFile(
      "drake/manipulation/models/ycb/sdf/004_sugar_box.sdf",
      math::RigidTransform<double>::Identity());

  // Check that the sugar box was added.
  EXPECT_EQ(station.get_multibody_plant().num_model_instances(),
            num_base_instances + 2);
}

// Check that making many stations does not exhaust resources.
GTEST_TEST(RlCitoStationTest, MultipleInstanceTest) {
  for (int i = 0; i < 20; ++i) {
    RlCitoStation<double> station;
    station.SetupCitoRlStation();
    station.Finalize();
  }
}

}  // namespace
}  // namespace rl_cito_station
}  // namespace examples
}  // namespace drake
