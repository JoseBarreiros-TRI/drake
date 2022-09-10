import argparse
from dataclasses import dataclass
import sys
import webbrowser

if sys.platform == "darwin":
    # TODO(jamiesnape): Fix this example on macOS Big Sur. Skipping on all
    # macOS for simplicity and because of the tendency for macOS versioning
    # schemes to unexpectedly change.
    # ImportError: C++ type is not registered in pybind:
    # NSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEE
    print("ERROR: Skipping this example on macOS because it fails on Big Sur")
    sys.exit(0)

import numpy as np

from pydrake.examples import (
    RlCitoStation, RlCitoStationHardwareInterface,
    CreateClutterClearingYcbObjectList)
from pydrake.geometry import DrakeVisualizer, Meshcat, MeshcatVisualizer
from pydrake.manipulation.planner import (
    DifferentialInverseKinematicsParameters)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (DiagramBuilder, LeafSystem,
                                       PublishEvent)
from pydrake.systems.lcm import LcmPublisherSystem
from pydrake.systems.primitives import FirstOrderLowPassFilter, VectorLogSink
from pydrake.systems.sensors import ImageToLcmImageArrayT, PixelType

from drake.examples.rl_cito_station.differential_ik import DifferentialIK

from drake import lcmt_image_array


class EndEffectorTeleop(LeafSystem):
    @dataclass
    class SliderDefault:
        """Default values for the meshcat sliders."""
        name: str
        """The name that is used to add / query values from."""
        default: float
        """The initial value of the slider."""

    _ROLL = SliderDefault("Roll", 0.0)
    _PITCH = SliderDefault("Pitch", 0.0)
    _YAW = SliderDefault("Yaw", 1.57)
    _X = SliderDefault("X", 0.0)
    _Y = SliderDefault("Y", 0.0)
    _Z = SliderDefault("Z", 0.0)

    def __init__(self, meshcat):
        """
        @param meshcat The already created pydrake.geometry.Meshcat instance.

        """

        LeafSystem.__init__(self)
        self.DeclareVectorOutputPort("rpy_xyz", 6, self.DoCalcOutput)
        self.meshcat = meshcat

        # Rotation control sliders.
        self.meshcat.AddSlider(
            name=self._ROLL.name, min=-2.0 * np.pi, max=2.0 * np.pi, step=0.01,
            value=self._ROLL.default)

        self.meshcat.AddSlider(
            name=self._PITCH.name, min=-2.0 * np.pi, max=2.0 * np.pi,
            step=0.01, value=self._PITCH.default)
        self.meshcat.AddSlider(
            name=self._YAW.name, min=-2.0 * np.pi, max=2.0 * np.pi,
            step=0.01, value=self._YAW.default)

        # Position control sliders.
        self.meshcat.AddSlider(
            name=self._X.name, min=-0.6, max=0.8, step=0.01,
            value=self._X.default)

        self.meshcat.AddSlider(
            name=self._Y.name, min=-0.8, max=0.3, step=0.01,
            value=self._Y.default)
        self.meshcat.AddSlider(
            name=self._Z.name, min=0.0, max=1.1, step=0.01,
            value=self._Z.default)

    def SetPose(self, pose):
        """
        @param pose is a RigidTransform or else any type accepted by
                    RigidTransform's constructor
        """
        tf = RigidTransform(pose)
        self.SetRPY(RollPitchYaw(tf.rotation()))
        self.SetXYZ(tf.translation())

    def SetRPY(self, rpy):
        """
        @param rpy is a RollPitchYaw object
        """
        self.meshcat.SetSliderValue(self._ROLL.name, rpy.roll_angle())

        self.meshcat.SetSliderValue(self._PITCH.name, rpy.pitch_angle())
        self.meshcat.SetSliderValue(self._YAW.name, rpy.yaw_angle())

    def SetXYZ(self, xyz):
        """
        @param xyz is a 3 element vector of x, y, z.
        """
        self.meshcat.SetSliderValue(self._X.name, xyz[0])
        self.meshcat.SetSliderValue(self._Y.name, xyz[1])
        self.meshcat.SetSliderValue(self._Z.name, xyz[2])

    def DoCalcOutput(self, context, output):
        roll = self.meshcat.GetSliderValue(self._ROLL.name)

        pitch = self.meshcat.GetSliderValue(self._PITCH.name)
        yaw = self.meshcat.GetSliderValue(self._YAW.name)

        x = self.meshcat.GetSliderValue(self._X.name)

        y = self.meshcat.GetSliderValue(self._Y.name)
        z = self.meshcat.GetSliderValue(self._Z.name)

        output.SetAtIndex(0, roll)
        output.SetAtIndex(1, pitch)
        output.SetAtIndex(2, yaw)
        output.SetAtIndex(3, x)
        output.SetAtIndex(4, y)
        output.SetAtIndex(5, z)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target_realtime_rate", type=float, default=1.0,
        help="Desired rate relative to real time.  See documentation for "
             "Simulator::set_target_realtime_rate() for details.")
    parser.add_argument(
        "--duration", type=float, default=np.inf,
        help="Desired duration of the simulation in seconds.")
    parser.add_argument(
        "--hardware", action='store_true',
        help="Use the RlCitoStationHardwareInterface instead of an "
             "in-process simulation.")
    parser.add_argument(
        "--test", action='store_true',
        help="Disable opening the gui window for testing.")
    parser.add_argument(
        "--filter_time_const", type=float, default=0.1,
        help="Time constant for the first order low pass filter applied to"
             "the teleop commands")
    parser.add_argument(
        "--velocity_limit_factor", type=float, default=1.0,
        help="This value, typically between 0 and 1, further limits the "
             "iiwa14 joint velocities. It multiplies each of the seven "
             "pre-defined joint velocity limits. "
             "Note: The pre-defined velocity limits are specified by "
             "iiwa14_velocity_limits, found in this python file.")
    parser.add_argument(
        '--setup', type=str, default='cito_rl',
        help="The manipulation station setup to simulate. ",
        choices=['cito_rl'])
    parser.add_argument(
        "-w", "--open-window", dest="browser_new",
        action="store_const", const=1, default=None,
        help=(
            "Open the MeshCat display in a new browser window.  NOTE: the "
            "slider controls are available in the meshcat viewer by clicking "
            "'Open Controls' in the top-right corner."))
    args = parser.parse_args()

    builder = DiagramBuilder()

    # NOTE: the meshcat instance is always created in order to create the
    # teleop controls (orientation sliders and open/close gripper button). When
    # args.hardware is True, the meshcat server will *not* display robot
    # geometry, but it will contain the joint sliders and open/close gripper
    # button in the "Open Controls" tab in the top-right of the viewing server.
    meshcat = Meshcat()

    if args.hardware:
        station = builder.AddSystem(RlCitoStationHardwareInterface())
        station.Connect()
    else:
        station = builder.AddSystem(RlCitoStation())

        # Initializes the chosen station type.
        if args.setup == 'cito_rl':
            station.SetupCitoRlStation()
            station.AddManipulandFromFile(
                "drake/examples/rl_cito_station/models/"
                + "061_foam_brick.sdf",
                RigidTransform(RotationMatrix.Identity(), [0.6, 0, 0]),
                "brick")

        station.Finalize()

        # If using meshcat, don't render the cameras, since RgbdCamera
        # rendering only works with drake-visualizer. Without this check,
        # running this code in a docker container produces libGL errors.
        geometry_query_port = station.GetOutputPort("geometry_query")

        # Connect the meshcat visualizer.
        meshcat_visualizer = MeshcatVisualizer.AddToBuilder(
            builder=builder,
            query_object_port=geometry_query_port,
            meshcat=meshcat)



        # Connect and publish to drake visualizer.
        DrakeVisualizer.AddToBuilder(builder, geometry_query_port)


    if args.browser_new is not None:
        url = meshcat.web_url()
        webbrowser.open(url=url, new=args.browser_new)

    robot = station.get_controller_plant()
    params = DifferentialInverseKinematicsParameters(robot.num_positions(),
                                                     robot.num_velocities())

    time_step = 0.005
    params.set_timestep(time_step)
    # True velocity limits for the IIWA14 (in rad, rounded down to the first
    # decimal)
    iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])

    # Stay within a small fraction of those limits for this teleop demo.
    factor = args.velocity_limit_factor
    params.set_joint_velocity_limits((-factor*iiwa14_velocity_limits,
                                      factor*iiwa14_velocity_limits))
    differential_ik = builder.AddSystem(DifferentialIK(
        robot, robot.GetFrameByName("iiwa_link_7"), params, time_step))

    builder.Connect(differential_ik.GetOutputPort("joint_position_desired"),
                    station.GetInputPort("iiwa_position"))

    teleop = builder.AddSystem(EndEffectorTeleop(
        meshcat))
    filter = builder.AddSystem(
        FirstOrderLowPassFilter(time_constant=args.filter_time_const, size=6))

    builder.Connect(teleop.get_output_port(0), filter.get_input_port(0))
    builder.Connect(filter.get_output_port(0),
                    differential_ik.GetInputPort("rpy_xyz_desired"))

    # When in regression test mode, log our joint velocities to later check
    # that they were sufficiently quiet.
    num_iiwa_joints = station.num_iiwa_joints()
    if args.test:
        iiwa_velocities = builder.AddSystem(VectorLogSink(num_iiwa_joints))
        builder.Connect(station.GetOutputPort("iiwa_velocity_estimated"),
                        iiwa_velocities.get_input_port(0))
    else:
        iiwa_velocities = None

    diagram = builder.Build()
    simulator = Simulator(diagram)

    # This is important to avoid duplicate publishes to the hardware interface:
    simulator.set_publish_every_time_step(False)

    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context())

    station.GetInputPort("iiwa_feedforward_torque").FixValue(
        station_context, np.zeros(num_iiwa_joints))

    # If the diagram is only the hardware interface, then we must advance it a
    # little bit so that first LCM messages get processed. A simulated plant is
    # already publishing correct positions even without advancing, and indeed
    # we must not advance a simulated plant until the sliders and filters have
    # been initialized to match the plant.
    if args.hardware:
        simulator.AdvanceTo(1e-6)

    q0 = station.GetOutputPort("iiwa_position_measured").Eval(
        station_context)
    differential_ik.parameters.set_nominal_joint_position(q0)

    teleop.SetPose(differential_ik.ForwardKinematics(q0))
    filter.set_initial_output_value(
        diagram.GetMutableSubsystemContext(
            filter, simulator.get_mutable_context()),
        teleop.get_output_port(0).Eval(diagram.GetMutableSubsystemContext(
            teleop, simulator.get_mutable_context())))
    differential_ik.SetPositions(diagram.GetMutableSubsystemContext(
        differential_ik, simulator.get_mutable_context()), q0)

    simulator.set_target_realtime_rate(args.target_realtime_rate)
    simulator.AdvanceTo(args.duration)

    # Ensure that our initialization logic was correct, by inspecting our
    # logged joint velocities.
    if args.test:
        iiwa_velocities_log = iiwa_velocities.FindLog(simulator.get_context())
        for time, qdot in zip(iiwa_velocities_log.sample_times(),
                              iiwa_velocities_log.data().transpose()):
            # TODO(jwnimmer-tri) We should be able to do better than a 40
            # rad/sec limit, but that's the best we can enforce for now.
            if qdot.max() > 0.1:
                print(f"ERROR: large qdot {qdot} at time {time}")
                sys.exit(1)


if __name__ == '__main__':
    main()
