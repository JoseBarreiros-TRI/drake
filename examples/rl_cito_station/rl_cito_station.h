#pragma once

#include <list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/geometry/render/render_engine.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/diagram.h"

namespace drake {
namespace examples {
namespace rl_cito_station {

/// Determines which sdf is loaded for the IIWA in the RlCitoStation.
enum class IiwaCollisionModel_ { kNoCollision, kBoxCollision };

/// Determines which schunk model is used for the RlCitoStation.
/// - kBox loads a model with a box collision geometry. This model is for those
///   who want simplified collision behavior.
/// - kBoxPlusFingertipSpheres loads a Schunk model with collision
///   spheres that models the indentations at tip of the fingers, in addition
///   to the box collision geometry on the fingers.
enum class SchunkCollisionModel { kBox, kBoxPlusFingertipSpheres };

/// Determines which manipulation station is simulated.
enum class Setup { kNone, kCitoRl};

/// @defgroup rl_cito_station_systems Manipulation Station
/// @{
/// @brief Systems related to the "manipulation station" used in the <a
/// href="https://manipulation.csail.mit.edu">MIT Intelligent Robot
/// Manipulation</a> class.
/// @ingroup example_systems
/// @}

/// A system that represents the complete manipulation station, including
/// exactly one robotic arm (a Kuka IIWA LWR), 
/// and anything a user might want to load into the model.
/// SetupDefaultStation() provides the setup that is used in the MIT
/// Intelligent Robot Manipulation class, which includes the supporting
/// structure for IIWA.  Alternative Setup___()
/// methods are provided, as well.
///
/// @system
/// name: RlCitoStation
/// input_ports:
/// - iiwa_position
/// - iiwa_feedforward_torque (optional)
/// output_ports:
/// - iiwa_position_commanded
/// - iiwa_position_measured
/// - iiwa_velocity_estimated
/// - iiwa_state_estimated
/// - iiwa_torque_commanded
/// - iiwa_torque_measured
/// - iiwa_torque_external

/// - <b style="color:orange">query_object</b>
/// - <b style="color:orange">contact_results</b>
/// - <b style="color:orange">plant_continuous_state</b>
/// - <b style="color:orange">geometry_poses</b>
/// @endsystem
///
/// Each pixel in the output image from `depth_image` is a 16bit unsigned
/// short in millimeters.
///
/// Note that outputs in <b style="color:orange">orange</b> are
/// available in the simulation, but not on the real robot.  The distinction
/// between q_measured and v_estimated is because the Kuka FRI reports
/// positions directly, but we have estimated v in our code that wraps the
/// FRI.
///
/// Consider the robot dynamics
///   M(q)vdot + C(q,v)v = τ_g(q) + τ_commanded + τ_joint_friction + τ_external,
/// where q == position, v == velocity, and τ == torque.
///
/// This model of the IIWA internal controller in the FRI software's
/// `JointImpedanceControlMode` is:
/// <pre>
///   τ_commanded = Mₑ(qₑ)vdot_desired + Cₑ(qₑ, vₑ)vₑ - τₑ_g(q) -
///                 τₑ_joint_friction + τ_feedforward
///   vdot_desired = PID(q_commanded, qₑ, v_commanded, vₑ)
/// </pre>
/// where Mₑ, Cₑ, τₑ_g, and τₑ_friction terms are now (Kuka's) estimates of the
/// true model, qₑ and vₑ are measured/estimation, and v_commanded
/// must be obtained from an online (causal) derivative of q_commanded.  The
/// result is
/// <pre>
///   M(q)vdot ≈ Mₑ(q)vdot_desired + τ_feedforward + τ_external,
/// </pre>
/// where the "approximately equal" comes from the differences due to the
/// estimated model/state.
///
/// The model implemented in this System assumes that M, C, and τ_friction
/// terms are perfect (except that they contain only a lumped mass
/// approximation of the gripper), and that the measured signals are
/// noise/bias free (e.g. q_measured = q, v_estimated = v, τ_measured =
/// τ_commanded).  What remains for τ_external is the generalized forces due
/// to contact (note that they could also include the missing contributions
/// from the gripper fingers, which the controller assumes are welded).
/// @see lcmt_iiwa_status.lcm for additional details/documentation.
///
///
/// To add objects into the environment for the robot to manipulate, use,
/// e.g.:
/// @code
/// RlCitoStation<double> station;
/// Parser parser(&station.get_mutable_multibody_plant(),
///                &station.get_mutable_scene_graph());
/// parser.AddModelFromFile("my.sdf", "my_model");
/// ...
/// // coming soon -- sugar API for adding additional objects.
/// station.Finalize()
/// @endcode
/// Note that you *must* call Finalize() before you can use this class as a
/// System.
///
/// @tparam_double_only
/// @ingroup rl_cito_station_systems
template <typename T>
class RlCitoStation : public systems::Diagram<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(RlCitoStation)

  /// Construct the EMPTY station model.
  ///
  /// @param time_step The time step used by MultibodyPlant<T>, and by the
  ///   discrete derivative used to approximate velocity from the position
  ///   command inputs.
  explicit RlCitoStation(double time_step = 0.002,  std::string contact_model = "point",  std::string contact_solver = "tamsi");

  /// Adds a default iiwa, RegisterIiwaControllerModel() with
  /// the appropriate arguments.
  /// @note Must be called before Finalize().
  /// @note Only one of the `Setup___()` methods should be called.
  /// @param collision_model Determines which sdf is loaded for the IIWA.
  /// @param schunk_model Determines which sdf is loaded for the Schunk.
  void SetupCitoRlStation(
    IiwaCollisionModel_ collision_model = IiwaCollisionModel_::kNoCollision);

  void SetDefaultState(const systems::Context<T>& station_context,
                       systems::State<T>* state) const override;

  /// Sets a random State for the chosen setup.
  /// @param context A const reference to the RlCitoStation context.
  /// @param state A pointer to the State of the RlCitoStation system.
  /// @param generator is the random number generator.
  /// @pre `state` must be the systems::State<T> object contained in
  /// `station_context`.
  void SetRandomState(const systems::Context<T>& station_context,
                      systems::State<T>* state,
                      RandomGenerator* generator) const override;

  /// Notifies the RlCitoStation that the IIWA robot model instance can
  /// be identified by @p iiwa_instance as well as necessary information to
  /// reload model for the internal controller's use. Assumes @p iiwa_instance
  /// has already been added to the MultibodyPlant.
  /// Note, the current implementation only allows @p parent_frame to be the
  /// world frame. The IIWA frame needs to directly contain @p child_frame.
  /// Only call this with custom IIWA models (i.e. not calling
  /// SetupDefaultStation()). Must be called before Finalize().
  /// @param model_path Full path to the model file.
  /// @param iiwa_instance Identifies the IIWA model.
  /// @param parent_frame Identifies frame P (the parent frame) in the
  /// MultibodyPlant that the IIWA model has been attached to.
  /// @param child_frame_name Identifies frame C (the child frame) in the IIWA
  /// model that is welded to frame P.
  /// @param X_PC Transformation between frame P and C.
  /// @throws If @p parent_frame is not the world frame.
  // TODO(siyuan.feng@tri.global): throws meaningful errors earlier here,
  // rather than in Finalize() if the arguments are inconsistent with the plant.
  // TODO(siyuan.feng@tri.global): remove the assumption that parent frame has
  // to be world.
  // TODO(siyuan.feng@tri.global): Some of these information should be
  // retrievable from the MultibodyPlant directly or MultibodyPlant should
  // provide partial tree cloning.
  void RegisterIiwaControllerModel(
      const std::string& model_path,
      const multibody::ModelInstanceIndex iiwa_instance,
      const multibody::Frame<T>& parent_frame,
      const multibody::Frame<T>& child_frame,
      const math::RigidTransform<double>& X_PC);

  /// Adds a single object for the robot to manipulate
  /// @note Must be called before Finalize().
  /// @param model_file The path to the .sdf model file of the object.
  /// @param X_WObject The pose of the object in world frame.
  void AddManipulandFromFile(const std::string& model_file,
                             const math::RigidTransform<double>& X_WObject,
                             const std::string manipuland_name={});

  // TODO(russt): Add scalar copy constructor etc once we support more
  // scalar types than T=double.  See #9573.

  /// Users *must* call Finalize() after making any additions to the
  /// multibody plant and before using this class in the Systems framework.
  /// This should be called exactly once.
  /// This assumes an IIWA have been added to the MultibodyPlant, and
  /// RegisterIiwaControllerModel() have been
  /// called.
  ///
  /// @see multibody::MultibodyPlant<T>::Finalize()
  void Finalize();

  /// Returns a reference to the main plant responsible for the dynamics of
  /// the robot and the environment.  This can be used to, e.g., add
  /// additional elements into the world before calling Finalize().
  const multibody::MultibodyPlant<T>& get_multibody_plant() const {
    return *plant_;
  }

  /// Returns a mutable reference to the main plant responsible for the
  /// dynamics of the robot and the environment.  This can be used to, e.g.,
  /// add additional elements into the world before calling Finalize().
  multibody::MultibodyPlant<T>& get_mutable_multibody_plant() {
    return *plant_;
  }

  /// Returns a reference to the SceneGraph responsible for all of the geometry
  /// for the robot and the environment.  This can be used to, e.g., add
  /// additional elements into the world before calling Finalize().
  const geometry::SceneGraph<T>& get_scene_graph() const {
    return *scene_graph_;
  }

  /// Returns a mutable reference to the SceneGraph responsible for all of the
  /// geometry for the robot and the environment.  This can be used to, e.g.,
  /// add additional elements into the world before calling Finalize().
  geometry::SceneGraph<T>& get_mutable_scene_graph() { return *scene_graph_; }

  /// Returns the name of the station's default renderer.
  static std::string default_renderer_name() { return default_renderer_name_; }

  /// Return a reference to the plant used by the inverse dynamics controller
  /// (which contains only a model of the iiwa + equivalent mass of the
  /// gripper).
  const multibody::MultibodyPlant<T>& get_controller_plant() const {
    return *owned_controller_plant_;
  }

  /// Gets the number of joints in the IIWA (only -- does not include the
  /// gripper).
  /// @pre must call one of the "setup" methods first to register an IIWA
  /// model.
  int num_iiwa_joints() const;

  /// Convenience method for getting all of the joint angles of the Kuka IIWA.
  /// This does not include the gripper.
  VectorX<T> GetIiwaPosition(const systems::Context<T>& station_context) const;

  /// Convenience method for setting all of the joint angles of the Kuka IIWA.
  /// Also sets the position history in the velocity command generator.
  /// @p q must have size num_iiwa_joints().
  /// @pre `state` must be the systems::State<T> object contained in
  /// `station_context`.
  void SetIiwaPosition(const systems::Context<T>& station_context,
                       systems::State<T>* state,
                       const Eigen::Ref<const VectorX<T>>& q) const;

  /// Convenience method for setting all of the joint angles of the Kuka IIWA.
  /// Also sets the position history in the velocity command generator.
  /// @p q must have size num_iiwa_joints().
  void SetIiwaPosition(systems::Context<T>* station_context,
                       const Eigen::Ref<const VectorX<T>>& q) const {
    SetIiwaPosition(*station_context, &station_context->get_mutable_state(), q);
  }

  /// Convenience method for getting all of the joint velocities of the Kuka
  // IIWA.  This does not include the gripper.
  VectorX<T> GetIiwaVelocity(const systems::Context<T>& station_context) const;

  /// Convenience method for setting all of the joint velocities of the Kuka
  /// IIWA. @v must have size num_iiwa_joints().
  /// @pre `state` must be the systems::State<T> object contained in
  /// `station_context`.
  void SetIiwaVelocity(const systems::Context<T>& station_context,
                       systems::State<T>* state,
                       const Eigen::Ref<const VectorX<T>>& v) const;

  /// Convenience method for setting all of the joint velocities of the Kuka
  /// IIWA. @v must have size num_iiwa_joints().
  void SetIiwaVelocity(systems::Context<T>* station_context,
                       const Eigen::Ref<const VectorX<T>>& v) const {
    SetIiwaVelocity(*station_context, &station_context->get_mutable_state(), v);
  }

  /// Set the position gains for the IIWA controller.
  /// @throws std::exception if Finalize() has been called.
  void SetIiwaPositionGains(const VectorX<double>& kp) {
    DRAKE_THROW_UNLESS(!plant_->is_finalized());
    iiwa_kp_ = kp;
  }

  /// Set the velocity gains for the IIWA controller.
  /// @throws std::exception if Finalize() has been called.
  void SetIiwaVelocityGains(const VectorX<double>& kd) {
    DRAKE_THROW_UNLESS(!plant_->is_finalized());
    iiwa_kd_ = kd;
  }

  /// Set the integral gains for the IIWA controller.
  /// @throws std::exception if Finalize() has been called.
  void SetIiwaIntegralGains(const VectorX<double>& ki) {
    DRAKE_THROW_UNLESS(!plant_->is_finalized());
    iiwa_ki_ = ki;
  }

 private:
  // Struct defined to store information about the how to parse and add a model.
  struct ModelInformation {
    /// This needs to have the full path. i.e. drake::FindResourceOrThrow(...)
    std::string model_path;
    multibody::ModelInstanceIndex model_instance;
    const multibody::Frame<T>* parent_frame{};
    const multibody::Frame<T>* child_frame{};
    math::RigidTransform<double> X_PC{math::RigidTransform<double>::Identity()};
  };

  // Assumes iiwa_model_info_ have already being populated.
  // Should only be called from Finalize().
  void MakeIiwaControllerModel();

  void AddDefaultIiwa(const IiwaCollisionModel_ collision_model);

  // These are only valid until Finalize() is called.
  std::unique_ptr<multibody::MultibodyPlant<T>> owned_plant_;
  std::unique_ptr<geometry::SceneGraph<T>> owned_scene_graph_;

  // These are valid for the lifetime of this system.
  std::unique_ptr<multibody::MultibodyPlant<T>> owned_controller_plant_;
  multibody::MultibodyPlant<T>* plant_;
  geometry::SceneGraph<T>* scene_graph_;
  static constexpr const char* default_renderer_name_ =
      "manip_station_renderer";

  // Populated by RegisterIiwaControllerModel() and
  ModelInformation iiwa_model_;

  // Store references to objects as *body* indices instead of model indices,
  // because this is needed for MultibodyPlant::SetFreeBodyPose(), etc.
  std::vector<multibody::BodyIndex> object_ids_;
  std::vector<math::RigidTransform<T>> object_poses_;

  // These are kp and kd gains for iiwa controllers.
  VectorX<double> iiwa_kp_;
  VectorX<double> iiwa_kd_;
  VectorX<double> iiwa_ki_;

  // Represents the manipulation station to simulate. This gets set in the
  // corresponding station setup function (e.g.,
  // SetupManipulationClassStation()), and informs how SetDefaultState()
  // initializes the sim.
  Setup setup_{Setup::kNone};
};

}  // namespace rl_cito_station
}  // namespace examples
}  // namespace drake
