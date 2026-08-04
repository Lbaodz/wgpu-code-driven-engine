use std::collections::HashMap;

use pub_fields::pub_fields;
use rapier3d::{
    control::KinematicCharacterController,
    dynamics::{
        CCDSolver, ImpulseJointHandle, ImpulseJointSet, IntegrationParameters, IslandManager,
        MultibodyJointSet, RigidBodyHandle, RigidBodySet,
    },
    geometry::{BroadPhaseBvh, ColliderSet, NarrowPhase},
    math::Vec3,
    pipeline::PhysicsPipeline,
};

const PHYSICS_RATE: f32 = 1.0 / 60.0;

#[pub_fields]
pub struct DoorAndJoint {
    door_handle: RigidBodyHandle,
    joint_handle: ImpulseJointHandle,
}

#[pub_fields]
#[derive(Default)]
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
    door_joint_handles: HashMap<u32, DoorAndJoint>,
    last_time: f32,
}

impl Collision {
    pub fn update_physics(&mut self, dt: f32) {
        self.last_time += dt;
        println!("{}", self.last_time);
        while self.last_time > PHYSICS_RATE {
            self.physics_pipeline.step(
                self.gravity,
                &self.integration,
                &mut self.island_manager,
                &mut self.broad_phasebvh,
                &mut self.narrow_phase,
                &mut self.rbs,
                &mut self.cs,
                &mut self.impulse_joint,
                &mut self.multi_body_joint,
                &mut self.ccd_solver,
                &(),
                &(),
            );
            self.last_time -= PHYSICS_RATE;
        };
        println!("last: {}", self.last_time);
    }
}
