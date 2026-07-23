use rapier3d::{
    control::KinematicCharacterController,
    dynamics::{
        CCDSolver, ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyJointSet,
        RigidBodySet, ImpulseJointHandle, RigidBodyHandle
    },
    geometry::{
        BroadPhaseBvh, ColliderSet, NarrowPhase,
    },
    math::Vec3,
    pipeline::{PhysicsPipeline},
};
use pub_fields::pub_fields;

#[pub_fields] 
pub struct Collision {
    rbs: RigidBodySet,
    cs: ColliderSet,
    char_handle: RigidBodyHandle,
    physics_pipeline: PhysicsPipeline,
    gravity: Vec3,
    integration: IntegrationParameters,
    island_manager: IslandManager,
    broad_phasebvh: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    ccd_solver: CCDSolver,
    impulse_joint: ImpulseJointSet,
    multi_body_joint: MultibodyJointSet,
    char_controller: KinematicCharacterController,
    doors_handle: Vec<RigidBodyHandle>,
    joints_handle: Vec<ImpulseJointHandle>,
}