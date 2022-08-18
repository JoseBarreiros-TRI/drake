#include "drake/examples/rl_cito_station/rl_cito_station.h"

#include <list>
#include <memory>
#include <string>
#include <utility>
#include <iostream>
#include "drake/common/find_resource.h"
#include "drake/geometry/render_vtk/factory.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/geometry/kinematics_vector.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/perception/depth_image_to_point_cloud.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/adder.h"
#include "drake/systems/primitives/constant_value_source.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/demultiplexer.h"
#include "drake/systems/primitives/discrete_derivative.h"
#include "drake/systems/primitives/linear_system.h"
#include "drake/systems/primitives/matrix_gain.h"
#include "drake/systems/primitives/pass_through.h"

namespace drake {
namespace examples {
namespace rl_cito_station {

using Eigen::Vector3d;
using Eigen::VectorXd;
using geometry::MakeRenderEngineVtk;
using geometry::RenderEngineVtkParams;
using geometry::SceneGraph;
using math::RigidTransform;
using math::RigidTransformd;
using math::RollPitchYaw;
using math::RotationMatrix;
using multibody::Joint;
using multibody::MultibodyPlant;
using multibody::PrismaticJoint;
using multibody::RevoluteJoint;
using multibody::SpatialInertia;
using multibody::ContactModel;
using multibody::DiscreteContactSolver;
using systems::Context;
using math::RollPitchYawd;

namespace internal {

template <typename T>

class ManipulandPoseExtractor final : public systems::LeafSystem<T> {
 public:
  explicit ManipulandPoseExtractor(const MultibodyPlant<T>* mbp)
      : mbp_(mbp) {
    input_ = &this->DeclareAbstractInputPort(
        "geometry_pose", Value<geometry::FramePoseVector<T>>());
    this->DeclareAbstractOutputPort(
        "output", &ManipulandPoseExtractor<T>::Extractor);
  }
  void Extractor(const Context<T>& context,
                 math::RigidTransform<T>* output) const {
    const geometry::FramePoseVector<T>& geometry_poses =
        input_->Eval<geometry::FramePoseVector<T>>(context);
    geometry::FrameId frame_id =
        mbp_->GetBodyFrameIdOrThrow(
            mbp_->GetBodyByName("base_link",
                               mbp_->GetModelInstanceByName("box")).index());
    *output = geometry_poses.value(frame_id);
  }

 private:
  const MultibodyPlant<T>* mbp_;
  // TODO(sammy-tri) Why in the heck can't I declare this as an InputPort<T>??
  const systems::InputPort<double>* input_{};
 
};

// Load a SDF model and weld it to the MultibodyPlant.
// @param model_path Full path to the sdf model file. i.e. with
// FindResourceOrThrow
// @param model_name Name of the added model instance.
// @param parent Frame P from the MultibodyPlant to which the new model is
// welded to.
// @param child_frame_name Defines frame C (the child frame), assumed to be
// present in the model being added.
// @param X_PC Transformation of frame C relative to frame P.
template <typename T>
multibody::ModelInstanceIndex AddAndWeldModelFrom(
    const std::string& model_path, const std::string& model_name,
    const multibody::Frame<T>& parent, const std::string& child_frame_name,
    const RigidTransform<double>& X_PC, MultibodyPlant<T>* plant) {
  DRAKE_THROW_UNLESS(!plant->HasModelInstanceNamed(model_name));

  multibody::Parser parser(plant);
  const multibody::ModelInstanceIndex new_model =
      parser.AddModelFromFile(model_path, model_name);
  const auto& child_frame = plant->GetFrameByName(child_frame_name, new_model);
  plant->WeldFrames(parent, child_frame, X_PC);
  return new_model;
}

}  // namespace internal

template <typename T>
RlCitoStation<T>::RlCitoStation(double time_step, std::string contact_model, std::string contact_solver)
    : owned_plant_(std::make_unique<MultibodyPlant<T>>(time_step)),
      owned_scene_graph_(std::make_unique<SceneGraph<T>>()),
      // Given the controller does not compute accelerations, it is irrelevant
      // whether the plant is continuous or discrete. We make it
      // discrete to avoid warnings about joint limits.
      owned_controller_plant_(std::make_unique<MultibodyPlant<T>>(1.0)) {
  // This class holds the unique_ptrs explicitly for plant and scene_graph
  // until Finalize() is called (when they are moved into the Diagram). Grab
  // the raw pointers, which should stay valid for the lifetime of the Diagram.
  plant_ = owned_plant_.get();
  scene_graph_ = owned_scene_graph_.get();
  plant_->RegisterAsSourceForSceneGraph(scene_graph_);
  scene_graph_->set_name("scene_graph");
  plant_->set_name("plant");

  // plant_->set_discrete_contact_solver(DiscreteContactSolver::kTamsi);
  //plant_->set_contact_model(ContactModel::kHydroelasticWithFallback);
  if (contact_solver == "sap") {
    plant_->set_discrete_contact_solver(DiscreteContactSolver::kSap);
  } else if (contact_solver == "tamsi") {
    plant_->set_discrete_contact_solver(DiscreteContactSolver::kTamsi);
  } else {
    throw std::runtime_error("Invalid contact solver '" + contact_solver +
                             "'.");
  }
  if (contact_model == "hydroelastic") {
    plant_->set_contact_model(ContactModel::kHydroelastic);
  } else if (contact_model == "point") {
    plant_->set_contact_model(ContactModel::kPoint);
  } else if (contact_model == "hydroelastic_with_fallback") {
    plant_->set_contact_model(ContactModel::kHydroelasticWithFallback);
  } else {
    throw std::runtime_error("Invalid contact model '" + contact_model +
                             "'.");
  }

  this->set_name("rl_cito_station");
}

template <typename T>
void RlCitoStation<T>::AddManipulandFromFile(
    const std::string& model_file, const RigidTransform<double>& X_WObject, std::string manipuland_name) {
  multibody::Parser parser(plant_);
  const auto model_index =
      parser.AddModelFromFile(FindResourceOrThrow(model_file),manipuland_name);
  const auto indices = plant_->GetBodyIndices(model_index);
  // Only support single-body objects for now.
  // Note: this could be generalized fairly easily... would just want to
  // set default/random positions for the non-floating-base elements below.
  DRAKE_DEMAND(indices.size() == 1);
  object_ids_.push_back(indices[0]);

  object_poses_.push_back(X_WObject);
}

template <typename T>
void RlCitoStation<T>::SetupCitoRlStation(
  IiwaCollisionModel_ collision_model) {

  DRAKE_DEMAND(setup_ == Setup::kNone);
  setup_ = Setup::kCitoRl;

  // Add the table and 80/20 workcell frame.
  {
    const double dx_table_center_to_robot_base = 1.2;
    const double dz_table_top_robot_base = 0;
    const std::string sdf_path = FindResourceOrThrow(
        "drake/examples/rl_cito_station/models/"
        "table.sdf");

    RigidTransform<double> X_WT(
        Vector3d(dx_table_center_to_robot_base, 0, dz_table_top_robot_base));
    internal::AddAndWeldModelFrom(sdf_path, "table", plant_->world_frame(),
                                  "table", X_WT, plant_);
  }

  AddDefaultIiwa(collision_model);

}

template <typename T>
int RlCitoStation<T>::num_iiwa_joints() const {
  DRAKE_DEMAND(iiwa_model_.model_instance.is_valid());
  return plant_->num_positions(iiwa_model_.model_instance);
}

template <typename T>
void RlCitoStation<T>::SetDefaultState(
    const systems::Context<T>& station_context,
    systems::State<T>* state) const {
  // Call the base class method, to initialize all systems in this diagram.
  systems::Diagram<T>::SetDefaultState(station_context, state);

  const auto& plant_context =
      this->GetSubsystemContext(*plant_, station_context);
  auto& plant_state = this->GetMutableSubsystemState(*plant_, state);

  DRAKE_DEMAND(object_ids_.size() == object_poses_.size());

  for (uint64_t i = 0; i < object_ids_.size(); i++) {
    plant_->SetFreeBodyPose(plant_context, &plant_state,
                            plant_->get_body(object_ids_[i]), object_poses_[i]);
  }

  // Use SetIiwaPosition to make sure the controller state is initialized to
  // the IIWA state.
  SetIiwaPosition(station_context, state, GetIiwaPosition(station_context));
  SetIiwaVelocity(station_context, state, VectorX<T>::Zero(num_iiwa_joints()));

}

template <typename T>
void RlCitoStation<T>::SetRandomState(
    const systems::Context<T>& station_context, systems::State<T>* state,
    RandomGenerator* generator) const {
  // Call the base class method, to initialize all systems in this diagram.
  systems::Diagram<T>::SetRandomState(station_context, state, generator);

  const auto& plant_context =
      this->GetSubsystemContext(*plant_, station_context);
  auto& plant_state = this->GetMutableSubsystemState(*plant_, state);

  // Separate the objects by lifting them up in z (in a random order).
  // TODO(russt): Replace this with an explicit projection into a statically
  // stable configuration.
  std::vector<multibody::BodyIndex> shuffled_object_ids(object_ids_);
  std::shuffle(shuffled_object_ids.begin(), shuffled_object_ids.end(),
               *generator);
  double z_offset = 0.05;
  for (const auto& body_index : shuffled_object_ids) {
    math::RigidTransform<T> pose =
        plant_->GetFreeBodyPose(plant_context, plant_->get_body(body_index));
    pose.set_translation(pose.translation() + Vector3d{0, 0, z_offset});
    // z_offset += 0.1;
    plant_->SetFreeBodyPose(plant_context, &plant_state,
                            plant_->get_body(body_index), pose);
  }

  // Use SetIiwaPosition to make sure the controller state is initialized to
  // the IIWA state.
  SetIiwaPosition(station_context, state, GetIiwaPosition(station_context));
  SetIiwaVelocity(station_context, state, VectorX<T>::Zero(num_iiwa_joints()));

}

template <typename T>
void RlCitoStation<T>::MakeIiwaControllerModel() {
  // Build the controller's version of the plant, which only contains the
  // IIWA and the equivalent inertia of the gripper.
  multibody::Parser parser(owned_controller_plant_.get());
  const auto controller_iiwa_model =
      parser.AddModelFromFile(iiwa_model_.model_path, "iiwa");

  owned_controller_plant_->WeldFrames(
      owned_controller_plant_->world_frame(),
      owned_controller_plant_->GetFrameByName(iiwa_model_.child_frame->name(),
                                              controller_iiwa_model),
      iiwa_model_.X_PC);

  owned_controller_plant_->set_name("controller_plant");
}

template <typename T>
void RlCitoStation<T>::Finalize() {
  DRAKE_THROW_UNLESS(iiwa_model_.model_instance.is_valid());

  MakeIiwaControllerModel();

  // Note: This deferred diagram construction method/workflow exists because we
  //   - cannot finalize plant until all of my objects are added, and
  //   - cannot wire up my diagram until we have finalized the plant.
  plant_->Finalize();

  // Set plant properties that must occur after finalizing the plant.
  VectorX<T> q0_iiwa(num_iiwa_joints());

  switch (setup_) {
    case Setup::kNone:
    case Setup::kCitoRl: {
      // Set the initial positions of the IIWA to a comfortable configuration
      // inside the workspace of the station.
      q0_iiwa << 0, 0.6, 0, -1.75, 0, 1.0, 0;

      std::uniform_real_distribution<symbolic::Expression> x(0.4, 0.65),
          y(-0.35, 0.35);
      const Vector3<symbolic::Expression> xyz{x(), y(), 0.0};
      std::uniform_real_distribution<symbolic::Expression> yaw(0.0, M_PI);
      math::RollPitchYaw<symbolic::Expression> rpy(0.0, 0.0, 0.0);

      //const math::RotationMatrix<double> X_WB_new(RollPitchYawd(0., 0., 0.4));
      for (const auto& body_index : object_ids_) {
        const multibody::Body<T>& body = plant_->get_body(body_index);
        plant_->SetFreeBodyRandomPositionDistribution(body, xyz);
        plant_->SetFreeBodyRandomRotationDistribution(body, rpy.ToRotationMatrix().ToQuaternion());
        //plant_->SetFreeBodyRandomRotationDistributionToUniform(body);
      }
      break;
    }
  }
  // Set the iiwa default configuration.
  const auto iiwa_joint_indices =
      plant_->GetJointIndices(iiwa_model_.model_instance);
  int q0_index = 0;
  for (const auto& joint_index : iiwa_joint_indices) {
    multibody::RevoluteJoint<T>* joint =
        dynamic_cast<multibody::RevoluteJoint<T>*>(
            &plant_->get_mutable_joint(joint_index));
    // Note: iiwa_joint_indices includes the WeldJoint at the base.  Only set
    // the RevoluteJoints.
    if (joint) {
      joint->set_default_angle(q0_iiwa[q0_index++]);
    }
  }

  systems::DiagramBuilder<T> builder;

  builder.AddSystem(std::move(owned_plant_));
  builder.AddSystem(std::move(owned_scene_graph_));

  builder.Connect(
      plant_->get_geometry_poses_output_port(),
      scene_graph_->get_source_pose_port(plant_->get_source_id().value()));
  builder.Connect(scene_graph_->get_query_output_port(),
                  plant_->get_geometry_query_input_port());

  const int num_iiwa_positions =
      plant_->num_positions(iiwa_model_.model_instance);
  DRAKE_THROW_UNLESS(num_iiwa_positions ==
                     plant_->num_velocities(iiwa_model_.model_instance));
  // Export the commanded positions via a PassThrough.
  auto iiwa_position =
      builder.template AddSystem<systems::PassThrough>(num_iiwa_positions);
  builder.ExportInput(iiwa_position->get_input_port(), "iiwa_position");
  builder.ExportOutput(iiwa_position->get_output_port(),
                       "iiwa_position_commanded");

  // Export iiwa "state" outputs.
  {
    auto demux = builder.template AddSystem<systems::Demultiplexer>(
        2 * num_iiwa_positions, num_iiwa_positions);
    builder.Connect(plant_->get_state_output_port(iiwa_model_.model_instance),
                    demux->get_input_port(0));
    builder.ExportOutput(demux->get_output_port(0), "iiwa_position_measured");
    builder.ExportOutput(demux->get_output_port(1), "iiwa_velocity_estimated");

    builder.ExportOutput(
        plant_->get_state_output_port(iiwa_model_.model_instance),
        "iiwa_state_estimated");
  }

  // Add the IIWA controller "stack".
  {
    owned_controller_plant_->Finalize();

    auto check_gains = [](const VectorX<double>& gains, int size) {
      return (gains.size() == size) && (gains.array() >= 0).all();
    };

    // Set default gains if.
    if (iiwa_kp_.size() == 0) {
      iiwa_kp_ = VectorXd::Constant(num_iiwa_positions, 100);
    }
    DRAKE_THROW_UNLESS(check_gains(iiwa_kp_, num_iiwa_positions));

    if (iiwa_kd_.size() == 0) {
      iiwa_kd_.resize(num_iiwa_positions);
      for (int i = 0; i < num_iiwa_positions; i++) {
        // Critical damping gains.
        iiwa_kd_[i] = 2 * std::sqrt(iiwa_kp_[i]);
      }
    }
    DRAKE_THROW_UNLESS(check_gains(iiwa_kd_, num_iiwa_positions));

    if (iiwa_ki_.size() == 0) {
      iiwa_ki_ = VectorXd::Constant(num_iiwa_positions, 1);
    }
    DRAKE_THROW_UNLESS(check_gains(iiwa_ki_, num_iiwa_positions));

    // Add the inverse dynamics controller.
    auto iiwa_controller = builder.template AddSystem<
        systems::controllers::InverseDynamicsController>(
        *owned_controller_plant_, iiwa_kp_, iiwa_ki_, iiwa_kd_, false);
    iiwa_controller->set_name("iiwa_controller");
    builder.Connect(plant_->get_state_output_port(iiwa_model_.model_instance),
                    iiwa_controller->get_input_port_estimated_state());

    // Add in feedforward torque.
    auto adder =
        builder.template AddSystem<systems::Adder>(2, num_iiwa_positions);
    builder.Connect(iiwa_controller->get_output_port_control(),
                    adder->get_input_port(0));
    // Use a passthrough to make the port optional.  (Will provide zero values
    // if not connected).
    auto torque_passthrough = builder.template AddSystem<systems::PassThrough>(
        Eigen::VectorXd::Zero(num_iiwa_positions));
    builder.Connect(torque_passthrough->get_output_port(),
                    adder->get_input_port(1));
    builder.ExportInput(torque_passthrough->get_input_port(),
                        "iiwa_feedforward_torque");
    builder.Connect(adder->get_output_port(), plant_->get_actuation_input_port(
                                                  iiwa_model_.model_instance));

    // Approximate desired state command from a discrete derivative of the
    // position command input port.
    auto desired_state_from_position = builder.template AddSystem<
        systems::StateInterpolatorWithDiscreteDerivative>(
            num_iiwa_positions, plant_->time_step(),
            true /* suppress_initial_transient */);
    desired_state_from_position->set_name("desired_state_from_position");
    builder.Connect(desired_state_from_position->get_output_port(),
                    iiwa_controller->get_input_port_desired_state());
    builder.Connect(iiwa_position->get_output_port(),
                    desired_state_from_position->get_input_port());

    // Export commanded torques:
    builder.ExportOutput(adder->get_output_port(), "iiwa_torque_commanded");
    builder.ExportOutput(adder->get_output_port(), "iiwa_torque_measured");
  }

  { 
    if (setup_ == Setup::kCitoRl){
      auto manipuland_pose_extractor=
          builder.template AddSystem<
            internal::ManipulandPoseExtractor<double>>(plant_);
      builder.Connect(
          plant_->get_geometry_poses_output_port(),
          manipuland_pose_extractor->get_input_port());
      builder.ExportOutput(manipuland_pose_extractor->get_output_port(),
                           "optitrack_manipuland_pose");
    }
  }

  builder.ExportOutput(plant_->get_generalized_contact_forces_output_port(
                           iiwa_model_.model_instance),
                       "iiwa_torque_external");

  builder.ExportOutput(scene_graph_->get_query_output_port(), "query_object");

  builder.ExportOutput(scene_graph_->get_query_output_port(),
                       "geometry_query");

  builder.ExportOutput(plant_->get_contact_results_output_port(),
                       "contact_results");
  builder.ExportOutput(plant_->get_state_output_port(),
                       "plant_continuous_state");
  // TODO(SeanCurtis-TRI) It seems with the scene graph query object port
  // exported, this output port is superfluous/undesirable. This port
  // contains the FramePoseVector that connects MBP to SG. Definitely better
  // to simply rely on the query object output port.
  builder.ExportOutput(plant_->get_geometry_poses_output_port(),
                       "geometry_poses");

  builder.BuildInto(this);
}

template <typename T>
VectorX<T> RlCitoStation<T>::GetIiwaPosition(
    const systems::Context<T>& station_context) const {
  const auto& plant_context =
      this->GetSubsystemContext(*plant_, station_context);
  return plant_->GetPositions(plant_context, iiwa_model_.model_instance);
}

template <typename T>
void RlCitoStation<T>::SetIiwaPosition(
    const drake::systems::Context<T>& station_context, systems::State<T>* state,
    const Eigen::Ref<const drake::VectorX<T>>& q) const {
  const int num_iiwa_positions =
      plant_->num_positions(iiwa_model_.model_instance);
  DRAKE_DEMAND(state != nullptr);
  DRAKE_DEMAND(q.size() == num_iiwa_positions);
  auto& plant_context = this->GetSubsystemContext(*plant_, station_context);
  auto& plant_state = this->GetMutableSubsystemState(*plant_, state);
  plant_->SetPositions(plant_context, &plant_state, iiwa_model_.model_instance,
                       q);
}

template <typename T>
VectorX<T> RlCitoStation<T>::GetIiwaVelocity(
    const systems::Context<T>& station_context) const {
  const auto& plant_context =
      this->GetSubsystemContext(*plant_, station_context);
  return plant_->GetVelocities(plant_context, iiwa_model_.model_instance);
}

template <typename T>
void RlCitoStation<T>::SetIiwaVelocity(
    const drake::systems::Context<T>& station_context, systems::State<T>* state,
    const Eigen::Ref<const drake::VectorX<T>>& v) const {
  const int num_iiwa_velocities =
      plant_->num_velocities(iiwa_model_.model_instance);
  DRAKE_DEMAND(state != nullptr);
  DRAKE_DEMAND(v.size() == num_iiwa_velocities);
  auto& plant_context = this->GetSubsystemContext(*plant_, station_context);
  auto& plant_state = this->GetMutableSubsystemState(*plant_, state);
  plant_->SetVelocities(plant_context, &plant_state, iiwa_model_.model_instance,
                        v);
}

template <typename T>
void RlCitoStation<T>::RegisterIiwaControllerModel(
    const std::string& model_path,
    const multibody::ModelInstanceIndex iiwa_instance,
    const multibody::Frame<T>& parent_frame,
    const multibody::Frame<T>& child_frame,
    const RigidTransform<double>& X_PC) {
  // TODO(siyuan.feng@tri.global): We really only just need to make sure
  // the parent frame is a AnchoredFrame(i.e. there is a rigid kinematic path
  // from it to the world), and record that X_WP. However, the computation to
  // query X_WP given a partially constructed plant is not feasible at the
  // moment, so we are forcing the parent frame to be the world instead.
  DRAKE_THROW_UNLESS(parent_frame.name() == plant_->world_frame().name());

  iiwa_model_.model_path = model_path;
  iiwa_model_.parent_frame = &parent_frame;
  iiwa_model_.child_frame = &child_frame;
  iiwa_model_.X_PC = X_PC;

  iiwa_model_.model_instance = iiwa_instance;
}

// Add default iiwa.
template <typename T>
void RlCitoStation<T>::AddDefaultIiwa(
    const IiwaCollisionModel_ collision_model) {
  std::string sdf_path;
  switch (collision_model) {
    case IiwaCollisionModel_::kNoCollision:
      sdf_path = FindResourceOrThrow(
          "drake/manipulation/models/iiwa_description/iiwa7/"
          "iiwa7_no_collision.sdf");
      break;
    case IiwaCollisionModel_::kBoxCollision:
      sdf_path = FindResourceOrThrow(
          "drake/manipulation/models/iiwa_description/iiwa7/"
          "iiwa7_with_box_collision.sdf");
      break;
  }
  const auto X_WI = RigidTransform<double>::Identity();
  auto iiwa_instance = internal::AddAndWeldModelFrom(
      sdf_path, "iiwa", plant_->world_frame(), "iiwa_link_0", X_WI, plant_);
  RegisterIiwaControllerModel(
      sdf_path, iiwa_instance, plant_->world_frame(),
      plant_->GetFrameByName("iiwa_link_0", iiwa_instance), X_WI);
}

}  // namespace rl_cito_station
}  // namespace examples
}  // namespace drake

// TODO(russt): Support at least NONSYMBOLIC_SCALARS.  See #9573.
//   (and don't forget to include default_scalars.h)
template class ::drake::examples::rl_cito_station::RlCitoStation<
    double>;
