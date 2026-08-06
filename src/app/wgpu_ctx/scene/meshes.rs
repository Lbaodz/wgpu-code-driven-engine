use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat};
use pub_fields::pub_fields;
use rapier3d::{
    control::{CharacterAutostep, CharacterLength, KinematicCharacterController},
    dynamics::{
        ImpulseJointHandle, ImpulseJointSet, RevoluteJointBuilder, RigidBodyBuilder,
        RigidBodyHandle, RigidBodySet,
    },
    geometry::{ColliderBuilder, ColliderSet, Group, InteractionGroups, InteractionTestMode},
    math::Vec3,
    pipeline::QueryFilter,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::FRAC_PI_2;
pub mod collision;

const DYN: Group = Group::GROUP_1;
const STA: Group = Group::GROUP_2;
const DOR: Group = Group::GROUP_3;

#[pub_fields]
#[repr(C)]
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Door {
    lock_pos: Vec3,
    lock_for_door: Vec3,
    scale: Vec3,
}

#[pub_fields]
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct IsDoor {
    id: Option<u32>,
    door: bool,
}

#[pub_fields]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct ModelMatrix {
    matrix: Mat4,
    pad: [u32; 48],
}

#[pub_fields]
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Deserialize, Serialize)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

#[pub_fields]
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Primitive {
    start: u32,
    count: u32,
    min: [f32; 3],
    max: [f32; 3],
    center: [f32; 3],
    extent: [f32; 3],
    texture_id: usize,
    offset_buffer: u32,
    is_door: IsDoor,
}

#[pub_fields]
#[derive(Debug)]
pub struct Texture {
    texture: wgpu::BindGroup,
}

#[pub_fields]
#[derive(Debug)]
pub struct Meshes {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    primitives: Vec<Primitive>,
    textures: Vec<Texture>,
    bind_group_matrices: wgpu::BindGroup,
    buffer_matrices: wgpu::Buffer,
    doors: HashMap<u32, Door>,
}

#[pub_fields]
#[repr(C)]
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BakedTexture {
    texture: Vec<u8>,
    width: u32,
    height: u32,
}

#[pub_fields]
#[repr(C)]
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BakedMeshes {
    vertex: Vec<Vertex>,
    index: Vec<u32>,
    primitives: Vec<Primitive>,
    baked_texture: Vec<BakedTexture>,
    matrices: Vec<Mat4>,
    doors: HashMap<u32, Door>,
}

impl collision::Collision {
    pub fn update_check_collision(&mut self, dt: f32, desire_movement: &Vec3, speed: f32) -> Vec3 {
        let char_data = &self.rbs[self.char_handle];
        let char_collider_handle = char_data.colliders()[0];
        let (character_shape, character_pos) = (
            self.cs[char_collider_handle].shared_shape().clone(),
            char_data.position(),
        );
        let query_pipeline = self.broad_phasebvh.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.rbs,
            &self.cs,
            QueryFilter::default().exclude_collider(char_collider_handle),
        );

        let mut collisions = Vec::new();
        let movement_result = self.char_controller.move_shape(
            dt,
            &query_pipeline,
            character_shape.as_ref(),
            character_pos,
            Vec3::new(desire_movement.x, desire_movement.y, desire_movement.z) * speed * dt,
            |collision| collisions.push(collision),
        );

        let mut query_pipeline_mut = self.broad_phasebvh.as_query_pipeline_mut(
            self.narrow_phase.query_dispatcher(),
            &mut self.rbs,
            &mut self.cs,
            QueryFilter::default().exclude_collider(char_collider_handle),
        );

        let mass = 50.0;
        self.char_controller.solve_character_collision_impulses(
            dt,
            &mut query_pipeline_mut,
            character_shape.as_ref(),
            mass,
            &collisions,
        );

        let rb = &mut self.rbs[self.char_handle];
        let new_pos = rb.position().translation + movement_result.translation;
        rb.set_next_kinematic_translation(new_pos);
        Vec3::new(new_pos.x, new_pos.y + 2.25, new_pos.z)
    }

    pub fn need_update_door(&self) -> bool {
        let mut should_check: Vec<bool> = Vec::new();
        let mut not_closed: bool = false;
        for (_, door_joint) in &self.door_joint_handles {
            let joint = &self
                .impulse_joint
                .get(door_joint.joint_handle)
                .expect("cant get joint from handle");
            let door = &self.rbs[joint.body2()];
            let door_rot = door.rotation();
            let angle = door_rot.to_euler(rapier3d::glamx::EulerRot::XYZ);
            not_closed = angle.1.abs() > 0.01;
            let angvel_y = door.angvel().y;
            let open = angvel_y.abs() > 0.01;
            should_check.push(open);
        }
        should_check.into_iter().any(|s| s) || not_closed
    }

    pub fn update_door(&mut self, forward: &Vec3, id: u32) {
        let joint = &mut self
            .impulse_joint
            .get_mut(
                self.door_joint_handles
                    .get(&id)
                    .expect("no door from id")
                    .joint_handle,
                false,
            )
            .expect("no joint handle");
        let door = &mut self.rbs[joint.body2()];
        let door_rot = door.rotation();
        let angle = door_rot.to_euler(rapier3d::glamx::EulerRot::XYZ);
        let curr_angle = angle.1;
        let data = &mut joint.data;
        let angvel_y = door.angvel().y;
        let curr_angle_abs = curr_angle.abs();

        let door_forw = door_rot * Vec3::X;
        let dot = Vec3::new(door_forw.x, door_forw.y, door_forw.z).dot(*forward) * 0.5 + 0.5;
        let pos = dot < 0.5 || (angvel_y > 0.5 && angvel_y < 2.0);
        let neg = dot > 0.5 || (angvel_y < -0.5 && angvel_y > -2.0);

        // when closed + no force or force
        if curr_angle_abs < 0.0001 {
            if pos {
                data.set_limits(rapier3d::dynamics::JointAxis::AngX, [0.0, FRAC_PI_2]);
                door.wake_up(true);
                println!("pos: {angvel_y}");
            } else if neg {
                data.set_limits(rapier3d::dynamics::JointAxis::AngX, [-FRAC_PI_2, 0.0]);
                door.wake_up(true);
                println!("neg: {angvel_y}");
            } else {
                data.set_limits(rapier3d::dynamics::JointAxis::AngX, [-0.005, 0.005]);
                println!("neither: {angvel_y}");
            }
        } else {
            if door.is_sleeping() && (pos || neg) {
                door.wake_up(true);
            }
        }
    }

    pub fn new_door_min_max(&self, id: u128) -> ([f32; 3], [f32; 3]) {
        let mut min_max: ([f32; 3], [f32; 3]) = ([0.0; 3], [0.0; 3]);
        for (_, door_joint) in &self.door_joint_handles {
            let rb = &self.rbs[door_joint.door_handle];
            let col_handle = self.cs.get(rb.colliders()[0]).expect("no collider door");
            if col_handle.user_data == id {
                let aabb = col_handle.compute_aabb();
                min_max = (aabb.mins.into(), aabb.maxs.into());
                return min_max;
            }
        }
        min_max
    }

    pub fn update_matrix_door(
        &mut self,
        id: u32,
        doors: &HashMap<u32, Door>,
        primitive: &Primitive,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
    ) {
        if let Some(door_handle) = self.need_update_matrix_door(id) {
            let door = self.rbs.get_mut(door_handle).expect("no door from handle");
            door.wake_up(true);
            let door_pos = door.position();
            let translation = door_pos.translation;
            let rot = door_pos.rotation;
            let sca = doors.get(&id).expect(&id.to_string()).scale;
            let model_matrix = Mat4::from_scale_rotation_translation(
                Vec3::new(sca.x, sca.y, sca.z),
                Quat::from_xyzw(rot.x, rot.y, rot.z, rot.w),
                Vec3::new(translation.x, translation.y, translation.z),
            );

            queue.write_buffer(
                buffer,
                primitive.offset_buffer as u64,
                bytemuck::bytes_of(&model_matrix),
            );
        }
    }

    fn need_update_matrix_door(&self, id: u32) -> Option<RigidBodyHandle> {
        let door_rb_handle = self
            .door_joint_handles
            .get(&id)
            .expect("no door from id")
            .door_handle; // loaded indexed by file parser
        let char_rb_handle = self.char_handle;

        if let (Some(char_rb), Some(door_rb)) =
            (self.rbs.get(char_rb_handle), self.rbs.get(door_rb_handle))
        {
            if let (Some(&door_col), Some(&char_col)) =
                (door_rb.colliders().first(), char_rb.colliders().first())
            {
                let door_collider = self.cs.get(door_col).expect("no collider from handle");
                let y_vel = door_rb.angvel().y.abs() > 0.01;
                let y = door_collider
                    .rotation()
                    .to_euler(rapier3d::glamx::EulerRot::XYZ)
                    .1
                    .abs()
                    > 0.01;
                return if y_vel || y {
                    Some(door_rb_handle)
                } else if let Some(contact_pair) =
                    self.narrow_phase.contact_pair(door_col, char_col)
                {
                    if contact_pair.has_any_active_contact() {
                        Some(door_rb_handle)
                    } else {
                        None
                    }
                } else {
                    None
                };
            } else {
                None
            }
        } else {
            None
        }
    }
}

pub fn load_door_collider(
    primitive: &Primitive,
    rbs: &mut RigidBodySet,
    cs: &mut ColliderSet,
    doors: &HashMap<u32, Door>,
    id: usize,
    joints: &mut ImpulseJointSet,
) -> (RigidBodyHandle, ImpulseJointHandle, u32) {
    let door_group = InteractionGroups::new(DOR, DYN, InteractionTestMode::Or);
    let door_data = doors
        .get(&(id as u32))
        .expect(format!("no door from id {id}").as_str());
    println!("{id}, {}", primitive.is_door.id.unwrap());
    let extent = primitive.extent;
    let center = primitive.center;
    let door = RigidBodyBuilder::dynamic()
        .translation(Vec3::new(center[0], center[1], center[2]))
        .angular_damping(4.0)
        .linear_damping(1.0)
        .build();
    let door_handle = rbs.insert(door);
    let offset = 0.1;

    let collider =
        ColliderBuilder::cuboid(extent[0] + offset, extent[1] + offset, extent[2] + offset)
            .user_data(id as u128)
            .collision_groups(door_group)
            .density(5.0)
            .build();
    cs.insert_with_parent(collider, door_handle, rbs);

    let lp = door_data.lock_pos;
    let lock = RigidBodyBuilder::fixed().translation(lp).build();
    let lock_handle = rbs.insert(lock);

    let joint = RevoluteJointBuilder::new(Vec3::new(0.0, 1.0, 0.0))
        .local_anchor1(Vec3::ZERO)
        .local_anchor2(
            doors
                .get(&(id as u32))
                .expect("no door from id(1)")
                .lock_for_door,
        )
        .motor_position(0.0, 50.0, 6.0)
        .limits([-FRAC_PI_2, FRAC_PI_2]);

    let joint_handle = joints.insert(lock_handle, door_handle, joint, true);

    (door_handle, joint_handle, id as u32)
}

pub fn load_static_collider(primitive: &Primitive, rbs: &mut RigidBodySet, cs: &mut ColliderSet) {
    let static_group = InteractionGroups::new(STA, DYN, InteractionTestMode::Or);
    let extent = primitive.extent;
    let center = primitive.center;
    let rb = RigidBodyBuilder::fixed()
        .translation(Vec3::new(center[0], center[1], center[2]))
        .build();
    let offset = 0.1;

    let collider =
        ColliderBuilder::cuboid(extent[0] + offset, extent[1] + offset, extent[2] + offset)
            .collision_groups(static_group)
            .build();
    let rb_handle = rbs.insert(rb);
    cs.insert_with_parent(collider, rb_handle, rbs);
}

pub fn load_player_collision(
    pos: &[f32; 3],
    rbs: &mut RigidBodySet,
    cs: &mut ColliderSet,
) -> (RigidBodyHandle, KinematicCharacterController) {
    let dyn_group = InteractionGroups::new(DYN, STA, InteractionTestMode::Or);
    let rb = RigidBodyBuilder::kinematic_position_based()
        .translation(Vec3::new(pos[0], pos[1], pos[2]))
        .build();
    let rb_handle = rbs.insert(rb);

    let collider = ColliderBuilder::capsule_y(2.25, 0.3)
        .collision_groups(dyn_group)
        .build();
    cs.insert_with_parent(collider, rb_handle, rbs);
    let mut char_controller = KinematicCharacterController::default();
    char_controller.autostep = Some(CharacterAutostep {
        max_height: CharacterLength::Absolute(0.5),
        min_width: CharacterLength::Absolute(0.2),
        include_dynamic_bodies: true,
    });
    char_controller.snap_to_ground = Some(CharacterLength::Absolute(0.5));

    (rb_handle, char_controller)
}
