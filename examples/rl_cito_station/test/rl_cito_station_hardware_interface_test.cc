#include "drake/examples/rl_cito_station/rl_cito_station_hardware_interface.h"  // noqa

#include <gtest/gtest.h>

#include "drake/systems/sensors/image.h"

namespace drake {
namespace examples {
namespace rl_cito_station {
namespace {

using Eigen::VectorXd;
using systems::BasicVector;

GTEST_TEST(RlCitoStationHardwareInterfaceTest, CheckPorts) {
  const int kNumIiwaDofs = 7;
  RlCitoStationHardwareInterface station;
  auto context = station.CreateDefaultContext();

  // Check sizes and names of the input ports.
  station.GetInputPort("iiwa_position")
      .FixValue(context.get(), VectorXd::Zero(kNumIiwaDofs));
  station.GetInputPort("iiwa_feedforward_torque")
      .FixValue(context.get(), VectorXd::Zero(kNumIiwaDofs));

  // Check sizes and names of the output ports.
  EXPECT_EQ(station.GetOutputPort("iiwa_position_commanded")
                .template Eval<BasicVector<double>>(*context)
                .size(),
            kNumIiwaDofs);
  EXPECT_EQ(station.GetOutputPort("iiwa_position_measured")
                .Eval<BasicVector<double>>(*context)
                .size(),
            kNumIiwaDofs);
  EXPECT_EQ(station.GetOutputPort("iiwa_velocity_estimated")
                .Eval<BasicVector<double>>(*context)
                .size(),
            kNumIiwaDofs);
  EXPECT_EQ(station.GetOutputPort("iiwa_torque_commanded")
                .Eval<BasicVector<double>>(*context)
                .size(),
            kNumIiwaDofs);
  EXPECT_EQ(station.GetOutputPort("iiwa_torque_measured")
                .Eval<BasicVector<double>>(*context)
                .size(),
            kNumIiwaDofs);
  EXPECT_EQ(station.GetOutputPort("iiwa_torque_external")
                .Eval<BasicVector<double>>(*context)
                .size(),
            kNumIiwaDofs);

  // TODO(russt): Consider adding mock lcm tests.  But doing so right now would
  // require exposing DrakeLcmInterface when I've so far tried to hide it.
}

}  // namespace
}  // namespace rl_cito_station
}  // namespace examples
}  // namespace drake
