"""
This example provides a direct collocation-based trajectory planning method
given a desired joint pose for Kuka iiwa arm respecting position, velocity,
and effort limits.
"""
import time

import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.geometry import MeshcatVisualizer, SceneGraph, StartMeshcat
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.solvers import BoundingBoxConstraint, SnoptSolver
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.trajectory_optimization import DirectCollocation
from pydrake.trajectories import PiecewisePolynomial


class TrajectoryPlanner():
    def __init__(self,
                initial_pose,
                preview=False,
                position_limit_factor=0.75,
                velocity_limit_factor=0.75,
                effort_limit_factor=0.75,
                additional_constraints=None):
        # Parse input args
        self.q0 = initial_pose
        self.preview = preview
        self.position_limit_factor = min(max(0.5, position_limit_factor), 1.0)
        self.velocity_limit_factor = min(max(0.5, velocity_limit_factor), 1.0)
        self.effort_limit_factor = min(max(0.5, effort_limit_factor), 1.0)
        self.additional_constraints=additional_constraints
        # Create the environment
        builder = DiagramBuilder()
        self.plant, self.scene = AddMultibodyPlantSceneGraph(builder, 5e-2)
        self.build_environment(add_manipuland=False)
        if preview:
            # Add a meshcat visualizer
            MeshcatVisualizer.AddToBuilder(builder=builder, scene_graph=self.scene,
                                        meshcat=StartMeshcat())
        # Build the diagram
        self.diagram = builder.Build()
        # Create contexts
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(
            self.diagram_context)

        # Visualize the initial pose
        self.diagram_context.SetDiscreteState(
            np.hstack((self.q0[:7], np.zeros(self.plant.num_velocities()))))
        self.diagram.Publish(self.diagram_context)

    def plan_to_joint_pose(self, q_goal):
        # Plan a trajectory to the desired joint pose
        time, states = self.solve_dircol(q_goal=q_goal, num_time_samples=31)

        # Preview the planned trajectory
        if self.preview:
            input("\nPress Enter to preview the planned trajectory...\n")
            for i, t in enumerate(time):
                print(f"t = {t}")
                print(f"\tq = {states[:7, i]}")
                print(f"\tv = {states[7:, i]}")
                self.diagram_context.SetDiscreteState(states[:, i])
                self.diagram.Publish(self.diagram_context)
                if t != time[-1]:
                    input("Press Enter to continue...")
            print("The preview is completed\n")

        # Smoothen and return the position trajectory
        plan = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            time, states[:7, :], np.zeros(7), np.zeros(7))
        return plan

    def solve_dircol(self, q_goal, num_time_samples=11):
        # Build a continuous-time plant and a scene
        plant = MultibodyPlant(0)
        scene_graph = SceneGraph()
        plant.RegisterAsSourceForSceneGraph(scene_graph)
        # Load the iiwa model
        model_file = FindResourceOrThrow(
            "drake/manipulation/models/iiwa_description"
            "/sdf/iiwa14_no_collision.sdf"
        )
        Parser(plant).AddModelFromFile(model_file)
        # Attach the base to the world frame and complete the plant definition
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("iiwa_link_0"))
        plant.Finalize()
        # Create a context for the plant
        context = plant.CreateDefaultContext()

        # Set up the collocation problem
        dircol = DirectCollocation(
            system=plant,
            context=context,
            num_time_samples=num_time_samples,
            minimum_timestep=2e-2,
            maximum_timestep=1e-1,
            input_port_index=plant.get_actuation_input_port().get_index())

        # Get the associated mathematical program
        prog = dircol.prog()
        x = dircol.state()
        u = dircol.input()

        # # Enforce equally-divided time segments
        # dircol.AddEqualTimeIntervalsConstraints()

        # Specify the task
        x0 = np.hstack((self.q0[:7], np.zeros(7)))
        xf = np.hstack((q_goal, np.zeros(7)))
        # Add the relevant constraints
        prog.AddBoundingBoxConstraint(x0, x0, dircol.initial_state())
        prog.AddBoundingBoxConstraint(xf, xf, dircol.final_state())

        # Add joint position, velocity, and effort constraints
        dircol.AddConstraintToAllKnotPoints(
            BoundingBoxConstraint(
                self.position_limit_factor*plant.GetPositionLowerLimits(),
                self.position_limit_factor*plant.GetPositionUpperLimits()
            ),
            x[:7])
        dircol.AddConstraintToAllKnotPoints(
            BoundingBoxConstraint(
                self.velocity_limit_factor*plant.GetVelocityLowerLimits(),
                self.velocity_limit_factor*plant.GetVelocityUpperLimits()
            ),
            x[7:])
        dircol.AddConstraintToAllKnotPoints(
            BoundingBoxConstraint(
                self.effort_limit_factor*plant.GetEffortLowerLimits(),
                self.effort_limit_factor*plant.GetEffortUpperLimits()
            ),
            u)

        # Penalize the total time
        dircol.AddFinalCost(dircol.time())

        # Penalize the effort
        w_tau = 1
        if w_tau > 0:
            dircol.AddRunningCost(w_tau * u.dot(u))

        # Penalize the velocities
        w_vel = 0
        if w_vel > 0:
            dircol.AddRunningCost(w_vel * x[7:].dot(x[7:]))

        # Set an initial guess by interpolating between
        # the initial and final states
        initial_x_trajectory = PiecewisePolynomial.FirstOrderHold(
            [0., 2.], np.column_stack((x0, xf)))
        dircol.SetInitialTrajectory(
            PiecewisePolynomial(), initial_x_trajectory)

        if self.additional_constraints != None:
            for a_constraint in self.additional_constraints:
                a_constraint(prog, plant, context)

        # Solve the program
        solver = SnoptSolver()
        print("\nRunning trajectory optimization...\n")
        t_start = time.time()
        result = solver.Solve(prog)
        # Print result details
        print(f"\tSolver type: {result.get_solver_id().name()}")
        print(f"\tSolver took {time.time() - t_start} s")
        print(f"\tSuccess: {result.is_success()}")
        print(f"\tOptimal cost: {result.get_optimal_cost()}")

        # Sample and return the solution
        t = dircol.GetSampleTimes(result)
        x = dircol.GetStateSamples(result)
        return (t, x)

    def build_environment(self, add_manipuland=False):
        # Add a table (i.e., a box) and fix it to the world
        sdf_file = FindResourceOrThrow(
            "drake/examples/rl_cito_station/models/floor.sdf")
        Parser(self.plant, self.scene).AddModelFromFile(sdf_file)
        self.plant.WeldFrames(frame_on_parent_F=self.plant.world_frame(),
                              frame_on_child_M=self.plant.GetFrameByName(
                                  "floor"),
                              X_FM=RigidTransform(p=[1.5, 0.0, -2.5e-2]))
        # Add an iiwa and fix its base to the world
        sdf_file = FindResourceOrThrow(
            "drake/examples/rl_cito_station/models/iiwa14_point_end_effector.sdf")
        Parser(self.plant, self.scene).AddModelFromFile(sdf_file)
        self.plant.WeldFrames(frame_on_parent_F=self.plant.world_frame(),
                              frame_on_child_M=self.plant.GetFrameByName(
                                  "iiwa_link_0"),
                              X_FM=RigidTransform(p=[0.0, 0.0, 0.0]))
        # Add the manipuland
        if add_manipuland:
            sdf_file = FindResourceOrThrow(
                "drake/examples/rl_cito_station/models/custom_box.sdf")
            Parser(self.plant, self.scene).AddModelFromFile(sdf_file)
        # Finalize the plant
        self.plant.Finalize()
