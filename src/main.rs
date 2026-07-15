use bytemuck::{Pod, Zeroable};
use cgmath::{Deg, InnerSpace, Matrix4, Point3, Quaternion, Vector3, Vector4, dot};
use rapier3d::dynamics::{RevoluteJointBuilder, RigidBodyHandle};
use rapier3d::{
    control::{CharacterAutostep, CharacterLength, KinematicCharacterController},
    dynamics::{
        CCDSolver, ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyJointSet,
        RigidBodyBuilder, RigidBodySet,
    },
    geometry::{
        BroadPhaseBvh, ColliderBuilder, ColliderSet, Group, InteractionGroups, InteractionTestMode,
        NarrowPhase,
    },
    math::Vec3,
    pipeline::{PhysicsPipeline, QueryFilter},
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::vec;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::application::ApplicationHandler;
use winit::event::{KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};
mod helper;

const DYN: Group = Group::GROUP_1;
const STA: Group = Group::GROUP_2;
const DOR: Group = Group::GROUP_3;

// door struct
struct Door {
    lock_pos: Vec3,
    lock_for_door: Vec3,
    scale: Vector3<f32>,
}

struct IsDoor {
    id: Option<u32>,
    door: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ModelMatrix {
    matrix: [[f32; 4]; 4],
    pad: [u32; 48],
}

#[derive(Default)]
struct Plane {
    normal: [f32; 3],
    d: f32,
}

#[derive(Default)]
struct PerformanceState {
    low: bool,
    mid: bool,
    ok: bool,
    high: bool,
    very_high: bool,
    epic: bool,
}

impl PerformanceState {
    fn all_false(&self) -> bool {
        ![
            self.low,
            self.mid,
            self.ok,
            self.high,
            self.very_high,
            self.epic,
        ]
        .into_iter()
        .any(|x| x)
    }
}

struct UI {
    egui_ctx: egui::Context,
    egui_winit_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

enum GameState {
    Menu,
    Play,
    Settings,
    Exit,
}

struct Planes {
    planes: [Plane; 6],
}

impl Planes {
    fn build_plane_from_matrix4(matrix4: Matrix4<f32>) -> Self {
        let m = matrix4;
        let make_plane = |nx, ny, nz, d| {
            let length = Vector3::new(nx, ny, nz).magnitude();
            Plane {
                normal: [nx / length, ny / length, nz / length],
                d: d / length,
            }
        };
        Self {
            planes: [
                make_plane(m.x.w + m.x.x, m.y.w + m.y.x, m.z.w + m.z.x, m.w.w + m.w.x),
                make_plane(m.x.w - m.x.x, m.y.w - m.y.x, m.z.w - m.z.x, m.w.w - m.w.x),
                make_plane(m.x.w + m.x.y, m.y.w + m.y.y, m.z.w + m.z.y, m.w.w + m.w.y),
                make_plane(m.x.w - m.x.y, m.y.w - m.y.y, m.z.w - m.z.y, m.w.w - m.w.y),
                make_plane(m.x.w + m.x.z, m.y.w + m.y.z, m.z.w + m.z.z, m.w.w + m.w.z),
                make_plane(m.x.w - m.x.z, m.y.w - m.y.z, m.z.w - m.z.z, m.w.w - m.w.z),
            ],
        }
    }

    fn frustum_culling(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        for plane in &self.planes {
            let mut positive_normal: [f32; 3] = [0.0; 3];
            let normal = plane.normal;
            for i in 0..3 {
                if normal[i] >= 0.0 {
                    positive_normal[i] = max[i];
                } else {
                    positive_normal[i] = min[i];
                }
            }
            if dot::<Vector3<f32>>(normal.into(), positive_normal.into()) + plane.d < 0.0 {
                return false;
            }
        }
        true
    }
}

#[derive(Default)]
struct InputState {
    w: bool,
    s: bool,
    a: bool,
    d: bool,
    q: bool,
    e: bool,
    shift: bool,
}

struct Camera {
    eye: Point3<f32>,
    target: Point3<f32>,
    up: Vector3<f32>,
    aspect: f32,
    fov: f32,
    near: f32,
    far: f32,
    yaw: f32,
    pitch: f32,
    is_moving: bool,
    is_rotating: bool,
}

impl Camera {
    fn make_camera(&self) -> Matrix4<f32> {
        let view = Matrix4::look_at_rh(self.eye, self.target, self.up);
        let v_fov = 2.0 * ((self.fov.to_radians() / 2.0).tan() / self.aspect).atan();
        let proj = cgmath::perspective(Deg(v_fov.to_degrees()), self.aspect, self.near, self.far);
        let wgpu_matrix_correction = Matrix4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0,
        );
        wgpu_matrix_correction * proj * view
    }

    fn update_target(&mut self) -> Vector3<f32> {
        let rad_yaw = self.yaw.to_radians();
        let rad_pitch = self.pitch.to_radians();

        let new_vector = Vector3::new(
            rad_yaw.cos() * rad_pitch.cos(),
            rad_pitch.sin(),
            rad_yaw.sin() * rad_pitch.cos(),
        );
        self.target = self.eye + new_vector;
        new_vector
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UniformCamera {
    uniform: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

struct Primitive {
    start: u32,
    count: u32,
    min: [f32; 3],
    max: [f32; 3],
    center: Vector3<f32>,
    extent: Vector3<f32>,
    texture_id: usize,
    offset_buffer: u32,
    is_door: IsDoor,
}

struct Texture {
    texture: wgpu::BindGroup,
}

struct Meshes {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    primitives: Vec<Primitive>,
    textures: Vec<Texture>,
    bind_group_matrices: wgpu::BindGroup,
    buffer_matrices: wgpu::Buffer,
    doors: Vec<Door>,
}

struct Scene {
    meshes: Vec<Meshes>,
}

struct Collision {
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
}

impl Collision {
    fn update_check_collision(
        &mut self,
        dt: f32,
        desire_movement: &Vector3<f32>,
        speed: f32,
    ) -> Point3<f32> {
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
        Point3::new(new_pos.x, new_pos.y + 2.25, new_pos.z)
    }

    fn check_door(
        &self,
        doors: &Vec<Door>,
        primitive: &Primitive,
    ) -> Vec<(Option<Matrix4<f32>>, Option<u32>)> {
        let index = doors.len();
        (0..index)
            .map(|i| {
                let door_pos = self.rbs[self.doors_handle[i]].position();
                let translation = door_pos.translation;
                let rot = door_pos.rotation;
                let sca = doors[i].scale;
                let t = Matrix4::from_translation(Vector3::new(
                    translation.x,
                    translation.y,
                    translation.z,
                ));
                let r = Matrix4::from(Quaternion::new(rot.x, rot.y, rot.z, rot.w));
                let s = Matrix4::from_nonuniform_scale(sca.x, sca.y, sca.z);
                (Some(t * r * s), Some(primitive.offset_buffer))
            })
            .collect::<Vec<(Option<Matrix4<f32>>, Option<u32>)>>()
    }
}

struct WgpuCtx {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    camera_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    camera: Camera,
    camera_planes: Planes,
    scene: Scene,
    depth_view: wgpu::TextureView,
    collision: Collision,
    ui: UI,
    game_state: GameState,
    playing: bool,
    mouse_locked: bool,
    fps_state: PerformanceState,
    audio: helper::Audio,
}

fn make_depth_tt(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        label: Some("depth tt"),
        mip_level_count: 1,
        sample_count: 1,
        size: wgpu::Extent3d {
            height: config.height,
            width: config.width,
            depth_or_array_layers: 1,
        },
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[wgpu::TextureFormat::Depth32Float],
    })
}

#[derive(Default)]
struct MyApp {
    window: Option<Arc<Window>>,
    gpu_ctx: Option<WgpuCtx>,
    last_time: Option<Instant>,
    input: InputState,
    fps: f32,
    should_draw: bool,
}

impl MyApp {
    fn set_default_app(&mut self, app_title: &str, event_loop: &ActiveEventLoop) -> Arc<Window> {
        let win_attr = Window::default_attributes().with_title(app_title);
        Arc::new(
            event_loop
                .create_window(win_attr)
                .expect("window create failed!"),
        )
    }

    fn update_last_time(&mut self) -> f32 {
        if let (Some(win), Some(last_time)) = (&self.window, &self.last_time) {
            let desire_time = Duration::from_secs_f32(1.0 / self.fps);
            let time_eslaped = last_time.elapsed();
            if time_eslaped < desire_time {
                let sleep_time = desire_time - time_eslaped;
                std::thread::sleep(sleep_time);
            };
            let now = Instant::now();
            let dt = now.duration_since(*last_time).as_secs_f32();
            self.last_time = Some(now);
            win.request_redraw();
            dt
        } else {
            panic!("no win object")
        }
    }
}

fn load_door_collider(
    primitive: &Primitive,
    rbs: &mut RigidBodySet,
    cs: &mut ColliderSet,
    doors: &Vec<Door>,
    id: usize,
    joints: &mut ImpulseJointSet,
) -> RigidBodyHandle {
    let door_group = InteractionGroups::new(DOR, DYN, InteractionTestMode::Or);
    let extent = primitive.extent;
    let center = primitive.center;
    let door = RigidBodyBuilder::dynamic()
        .translation(Vec3::new(center.x, center.y, center.z))
        .build();
    let door_handle = rbs.insert(door);
    let offset = 0.1;

    let collider = ColliderBuilder::cuboid(extent.x + offset, extent.y + offset, extent.z + offset)
        .collision_groups(door_group)
        .build();
    cs.insert_with_parent(collider, door_handle, rbs);

    let lp = doors[id].lock_pos;
    let lock = RigidBodyBuilder::fixed().translation(lp).build();
    let lock_handle = rbs.insert(lock);

    let joint = RevoluteJointBuilder::new(Vec3::new(0.0, 1.0, 0.0))
        .local_anchor1(lp)
        .local_anchor2(doors[id].lock_for_door)
        .limits([0.0, std::f32::consts::FRAC_PI_2]);

    joints.insert(lock_handle, door_handle, joint, true);
    door_handle
}

fn load_static_collider(
    primitive: &Primitive,
    rbs: &mut RigidBodySet,
    cs: &mut ColliderSet,
    doors: &Vec<Door>,
    joints: &mut ImpulseJointSet,
) -> Option<RigidBodyHandle> {
    if primitive.is_door.door && !(doors.len() == 0) {
        if let Some(i) = primitive.is_door.id {
            let id = i as usize;
            return Some(load_door_collider(primitive, rbs, cs, doors, id, joints));
        } else {
            None
        }
    } else {
        let static_group = InteractionGroups::new(STA, DYN, InteractionTestMode::Or);
        let extent = primitive.extent;
        let center = primitive.center;
        let rb = RigidBodyBuilder::fixed()
            .translation(Vec3::new(center.x, center.y, center.z))
            .build();
        let offset = 0.1;

        let collider =
            ColliderBuilder::cuboid(extent.x + offset, extent.y + offset, extent.z + offset)
                .collision_groups(static_group)
                .build();
        let rb_handle = rbs.insert(rb);
        cs.insert_with_parent(collider, rb_handle, rbs);

        None
    }
}

fn load_player_collision(
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

fn convert_aabb_from_matrix(
    min: Vector3<f32>,
    max: Vector3<f32>,
    matrix: Matrix4<f32>,
) -> ([f32; 3], [f32; 3], Vector3<f32>, Vector3<f32>) {
    let old_extent = (max - min) * 0.5;
    let old_center = (min + max) * 0.5;
    let center4 = matrix * Vector4::new(old_center.x, old_center.y, old_center.z, 1.0);
    let center = Vector3::new(center4.x, center4.y, center4.z);

    let mut extent = Vector3::new(0.0, 0.0, 0.0);
    for i in 0..3 {
        for j in 0..3 {
            extent[i] += matrix[j][i] * old_extent[i];
        }
    }
    let min = center - extent;
    let max = center + extent;
    (min.into(), max.into(), center, extent)
}

fn load_model(
    path: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture_layout: &wgpu::BindGroupLayout,
    model_matrix_layout: &wgpu::BindGroupLayout,
) -> Vec<Meshes> {
    let (document, buffers, images) = gltf::import(path).expect("Not found path");
    // return
    let mut objects: Vec<Meshes> = Vec::new();
    // buffer render offset **IMPORTANT: NO SEPARATE BUFFER FOR EACH RETURN**
    let mut all_verticles: Vec<Vertex> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();
    // index range
    let mut pris: Vec<Primitive> = Vec::new();
    // tt
    let mut textures: Vec<Texture> = Vec::new();
    let mut img_cache: HashMap<usize, usize> = HashMap::new();
    // model matrices
    let mut matrices: Vec<ModelMatrix> = Vec::new();
    let mut offset_buffer: u32 = 0;
    // door check
    let mut door_pos: Vec<Vec3> = Vec::new();
    let mut lock_pos: Vec<Vec3> = Vec::new();
    let mut id: u32 = 0;
    let mut scales: Vec<Vector3<f32>> = Vec::new();

    for scene in document.scenes() {
        for node in scene.nodes() {
            // matrix model
            let mut door = false;
            let model_matrix: [[f32; 4]; 4] = match node.transform() {
                gltf::scene::Transform::Matrix { matrix } => matrix,
                gltf::scene::Transform::Decomposed {
                    translation,
                    rotation,
                    scale,
                } => {
                    match node.name() {
                        Some("door") => {
                            door = true;
                            door_pos.push(Vec3::new(
                                translation[0],
                                translation[1],
                                translation[2],
                            ));
                            scales.push(Vector3::new(scale[0], scale[1], scale[2]));
                            if !(id == 0) {
                                id += 1;
                            };
                        }
                        Some("empty") => {
                            lock_pos.push(Vec3::new(
                                translation[0],
                                translation[1],
                                translation[2],
                            ));
                        }
                        None => (),
                        _ => (),
                    };
                    let t: Matrix4<f32> = Matrix4::from_translation(Vector3::new(
                        translation[0],
                        translation[1],
                        translation[2],
                    ));
                    let r: Matrix4<f32> = Matrix4::from(Quaternion::new(
                        rotation[3],
                        rotation[0],
                        rotation[1],
                        rotation[2],
                    ));
                    let s: Matrix4<f32> =
                        Matrix4::from_nonuniform_scale(scale[0], scale[1], scale[2]);
                    (t * r * s).into()
                }
            };

            if let Some(mesh) = node.mesh() {
                for primitive in mesh.primitives() {
                    // min max
                    let accessor = primitive.get(&gltf::Semantic::Positions).unwrap();
                    let (mi, ma) = (accessor.min().unwrap(), accessor.max().unwrap());
                    let min_raw: [f32; 3] =
                        serde_json::from_value(mi.clone()).expect("cant extract min");
                    let max_raw: [f32; 3] =
                        serde_json::from_value(ma.clone()).expect("cant extract min");
                    let (min, max, center, extent) = convert_aabb_from_matrix(
                        min_raw.into(),
                        max_raw.into(),
                        model_matrix.into(),
                    );

                    // texture
                    let material = primitive.material();
                    let img_info = material.emissive_texture().expect("no info");
                    let img_index = img_info.texture().source().index();
                    let texture_id: usize = *img_cache.entry(img_index).or_insert_with(|| {
                        let img_data = &images[img_index];
                        let rgba_pixel = match img_data.format {
                            gltf::image::Format::R8G8B8 => {
                                let mut converter: Vec<u8> = Vec::with_capacity(
                                    img_data.width as usize * 4 * img_data.height as usize,
                                );
                                for chunk in img_data.pixels.chunks_exact(3) {
                                    converter.push(chunk[0]);
                                    converter.push(chunk[1]);
                                    converter.push(chunk[2]);
                                    converter.push(255);
                                }
                                converter
                            }
                            gltf::image::Format::R8G8B8A8 => img_data.pixels.clone(),
                            _ => {
                                panic!("unsupported format texture image")
                            }
                        };
                        let texture = device.create_texture(&wgpu::TextureDescriptor {
                            label: Some("tt"),
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba8UnormSrgb,
                            mip_level_count: 1,
                            sample_count: 1,
                            size: wgpu::Extent3d {
                                width: img_data.width,
                                height: img_data.height,
                                depth_or_array_layers: 1,
                            },
                            usage: wgpu::TextureUsages::TEXTURE_BINDING
                                | wgpu::TextureUsages::COPY_DST,
                            view_formats: &[],
                        });
                        queue.write_texture(
                            texture.as_image_copy(),
                            &rgba_pixel,
                            wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(img_data.width as u32 * 4),
                                rows_per_image: Some(img_data.height),
                            },
                            wgpu::Extent3d {
                                width: img_data.width,
                                height: img_data.height,
                                depth_or_array_layers: 1,
                            },
                        );
                        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                            address_mode_u: wgpu::AddressMode::Repeat,
                            address_mode_v: wgpu::AddressMode::Repeat,
                            address_mode_w: wgpu::AddressMode::Repeat,
                            ..Default::default()
                        });
                        let texture_bind_group =
                            device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("tt bindgroup"),
                                layout: &texture_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(&view),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::Sampler(&sampler),
                                    },
                                ],
                            });
                        textures.push(Texture {
                            texture: texture_bind_group,
                        });
                        textures.len() - 1
                    });

                    // primitive
                    let reader = primitive.reader(|b| Some(&buffers[b.index()]));
                    let pos: Vec<[f32; 3]> =
                        reader.read_positions().expect("cant get pos").collect();
                    let count = pos.len();
                    let uv: Vec<[f32; 2]> = reader
                        .read_tex_coords(0)
                        .expect("no uv")
                        .into_f32()
                        .map(|i| i)
                        .collect();
                    let nor: Vec<[f32; 3]> = reader.read_normals().expect("cant get nor").collect();

                    let verticle: Vec<Vertex> = (0..count)
                        .map(|i| Vertex {
                            position: pos[i],
                            normal: nor[i],
                            uv: uv[i],
                        })
                        .collect();

                    let offset = all_verticles.len() as u32;
                    let start = all_indices.len() as u32;
                    let indices: Vec<u32> = reader
                        .read_indices()
                        .unwrap()
                        .into_u32()
                        .map(|i| i + offset)
                        .collect();
                    println!("len: {:?}", matrices.len());
                    println!("{offset_buffer}");

                    let id = if door { Some(id) } else { None };
                    let is_door = IsDoor { id, door };

                    pris.push(Primitive {
                        start: start,
                        count: indices.len() as u32,
                        min,
                        max,
                        center,
                        extent,
                        texture_id,
                        offset_buffer,
                        is_door,
                    });
                    // matrices
                    matrices.push(ModelMatrix {
                        matrix: model_matrix,
                        pad: [0; 48],
                    });
                    offset_buffer = matrices.len() as u32 * 256;

                    all_verticles.extend(verticle);
                    all_indices.extend(indices);
                }
            }
        }
    }
    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("vx_buffer"),
        contents: bytemuck::cast_slice(&all_verticles),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("ix_buffer"),
        contents: bytemuck::cast_slice(&all_indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    // model matrices
    let buffer_matrices = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matrices buffer"),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        size: matrices.len() as u64 * 256,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer_matrices, 0, bytemuck::cast_slice(&matrices));
    let bind_group_matrices = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group_matrices"),
        layout: &model_matrix_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer_matrices,
                offset: 0,
                size: std::num::NonZero::new(64),
            }),
        }],
    });

    let mut doors: Vec<Door> = Vec::new();
    println!("{:?}, ANDANDAND {:?}", lock_pos, door_pos);
    if (lock_pos.len() == 0) || (door_pos.len() == 0) {
        ()
    } else {
        doors = (0..id)
            .map(|index| {
                let i = index as usize;
                let d = lock_pos[i] - door_pos[i];
                Door {
                    lock_pos: lock_pos[i],
                    lock_for_door: Vec3::new(d.x, d.y, d.z),
                    scale: scales[i],
                }
            })
            .collect();
    }

    objects.push(Meshes {
        vertex_buffer,
        index_buffer,
        primitives: pris,
        textures,
        bind_group_matrices,
        buffer_matrices,
        doors,
    });

    objects
}

// break line <------------------------------------------------------------------------------------>
impl WgpuCtx {
    fn get_frame_view(&self) -> Option<(wgpu::SurfaceTexture, wgpu::TextureView)> {
        let frame_enum = self.surface.get_current_texture();
        match frame_enum {
            wgpu::CurrentSurfaceTexture::Success(frame) => {
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Texture View"),
                    ..Default::default()
                });
                Some((frame, view))
            }
            _ => None,
        }
    }

    fn update_ui_menu(
        &mut self,
        win: &Window,
        mut encoder: &mut wgpu::CommandEncoder,
    ) -> (
        std::vec::Vec<egui::ClippedPrimitive>,
        egui_wgpu::ScreenDescriptor,
        std::vec::Vec<egui::TextureId>,
    ) {
        let paint_jobs;
        let screen_descriptor;
        let texture_ui;
        {
            let ui = &mut self.ui;
            let raw_input = ui.egui_winit_state.take_egui_input(&win);
            ui.egui_ctx.begin_pass(raw_input);
            egui::Window::new("ok")
                .title_bar(false)
                .resizable(true)
                .collapsible(false)
                .fixed_size(egui::vec2(1000.0, 1000.0))
                .frame(
                    egui::Frame::window(&ui.egui_ctx.global_style())
                        .fill(egui::Color32::from_rgba_unmultiplied(0, 0, 0, 0))
                        .stroke(egui::Stroke::new(
                            0.0,
                            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 0),
                        ))
                        .corner_radius(0)
                        .shadow(egui::epaint::Shadow {
                            offset: [0, 0],
                            blur: 0,
                            spread: 0,
                            color: egui::Color32::from_black_alpha(0),
                        }),
                )
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(-500.0, -300.0))
                .show(&ui.egui_ctx, |egui| {
                    egui.style_mut().interaction.selectable_labels = false;
                    egui.add_space(50.0);
                    egui.label(
                        egui::RichText::new("NoGAmE")
                            .font(egui::FontId::monospace(150.0))
                            .strong()
                            .color(egui::Color32::from_rgba_unmultiplied(210, 20, 20, 180)),
                    );
                    egui.add_space(150.0);

                    // play
                    if helper::layout_but_ui(egui, "Play", "play_but").clicked() {
                        self.game_state = GameState::Play;
                        self.playing = true;
                        win.set_cursor_visible(false);
                        let _ = win.set_cursor_grab(CursorGrabMode::Locked);
                        self.mouse_locked = true;
                    };
                    // set
                    if helper::layout_but_ui(egui, "Settings", "settings_but").clicked() {
                        self.game_state = GameState::Settings
                    };
                    egui.add_space(80.0);
                    // ext
                    if helper::layout_but_ui(egui, "Exit", "exit_but").clicked() {
                        self.game_state = GameState::Exit
                    };
                });

            let ui_data = ui.egui_ctx.end_pass();

            for (id, texture) in &ui_data.textures_delta.set {
                ui.egui_renderer
                    .update_texture(&self.device, &self.queue, *id, texture);
            }
            texture_ui = ui_data.textures_delta.free.clone();
            ui.egui_winit_state
                .handle_platform_output(&win, ui_data.platform_output);
            paint_jobs = ui
                .egui_ctx
                .tessellate(ui_data.shapes, ui_data.pixels_per_point);
            screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point: ui_data.pixels_per_point,
            };
            ui.egui_renderer.update_buffers(
                &self.device,
                &self.queue,
                &mut encoder,
                &paint_jobs,
                &screen_descriptor,
            );
        }
        (paint_jobs, screen_descriptor, texture_ui)
    }

    fn update_ui_settings(
        &mut self,
        win: &Window,
        mut encoder: &mut wgpu::CommandEncoder,
    ) -> (
        std::vec::Vec<egui::ClippedPrimitive>,
        egui_wgpu::ScreenDescriptor,
        std::vec::Vec<egui::TextureId>,
        Option<f32>,
    ) {
        let paint_jobs;
        let screen_descriptor;
        let texture_ui;
        let mut fps: f32 = 0.0;
        {
            let ui = &mut self.ui;
            let raw_input = ui.egui_winit_state.take_egui_input(&win);
            ui.egui_ctx.begin_pass(raw_input);
            egui::Window::new("ok")
                .title_bar(false)
                .resizable(true)
                .collapsible(false)
                .fixed_size(egui::vec2(1000.0, 1000.0))
                .frame(
                    egui::Frame::window(&ui.egui_ctx.global_style())
                        .fill(egui::Color32::from_rgba_unmultiplied(0, 0, 0, 0))
                        .stroke(egui::Stroke::new(
                            0.0,
                            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 0),
                        ))
                        .corner_radius(0)
                        .shadow(egui::epaint::Shadow {
                            offset: [0, 0],
                            blur: 0,
                            spread: 0,
                            color: egui::Color32::from_black_alpha(0),
                        }),
                )
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(-410.0, -300.0))
                .show(&ui.egui_ctx, |egui| {
                    egui.style_mut().interaction.selectable_labels = false;
                    egui.add_space(50.0);
                    egui.label(
                        egui::RichText::new("Settings")
                            .font(egui::FontId::monospace(150.0))
                            .strong()
                            .color(egui::Color32::from_rgba_unmultiplied(210, 20, 20, 180)),
                    );
                    egui.add_space(150.0);

                    // fov
                    let (fov, fov_val) =
                        helper::layout_sld_ui(egui, "Fov", &mut self.camera.fov, "fov", 30..120);
                    if fov.clicked() || fov.drag_stopped() {
                        self.camera.fov = fov_val;
                        self.camera.is_moving = true;
                    };
                    egui.add_space(5.0);
                    // sound (deadcode)
                    let (sound, sound_val) = helper::layout_sld_ui(
                        egui,
                        "Sound",
                        &mut self.audio.volume,
                        "sound",
                        0..100,
                    );
                    if sound.clicked() || sound.drag_stopped() {
                        self.audio.volume = sound_val;
                    };
                    egui.add_space(20.0);
                    // fps editor
                    egui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("FPS")
                                .color(egui::Color32::WHITE)
                                .font(egui::FontId::monospace(30.0)),
                        );
                        ui.add_space(80.0);
                        if helper::layout_chb_ui(ui, "30", "30_chb", &self.fps_state.low).clicked()
                        {
                            self.fps_state = PerformanceState {
                                low: !self.fps_state.low,
                                ..Default::default()
                            };
                            fps = 30.0;
                        };
                        if helper::layout_chb_ui(ui, "60", "60_chb", &self.fps_state.mid).clicked()
                        {
                            self.fps_state = PerformanceState {
                                mid: !self.fps_state.mid,
                                ..Default::default()
                            };
                            fps = 60.0;
                        };
                        if helper::layout_chb_ui(ui, "90", "90_chb", &self.fps_state.ok).clicked() {
                            self.fps_state = PerformanceState {
                                ok: !self.fps_state.ok,
                                ..Default::default()
                            };
                            fps = 90.0;
                        };
                        if helper::layout_chb_ui(ui, "120", "120_chb", &self.fps_state.high)
                            .clicked()
                        {
                            self.fps_state = PerformanceState {
                                high: !self.fps_state.high,
                                ..Default::default()
                            };
                            fps = 120.0;
                        };
                        if helper::layout_chb_ui(ui, "144", "144_chb", &self.fps_state.very_high)
                            .clicked()
                        {
                            self.fps_state = PerformanceState {
                                very_high: !self.fps_state.very_high,
                                ..Default::default()
                            };
                            fps = 144.0;
                        };
                        if helper::layout_chb_ui(ui, "240", "240_chb", &self.fps_state.epic)
                            .clicked()
                        {
                            self.fps_state = PerformanceState {
                                epic: !self.fps_state.epic,
                                ..Default::default()
                            };
                            fps = 240.0;
                        };
                        if self.fps_state.all_false() {
                            if helper::layout_chb_ui(ui, "???", "???", &true).clicked() {
                                ()
                            };
                        };
                    });
                    egui.add_space(40.0);
                    // ext
                    if helper::layout_but_ui(egui, "Back", "back_but").clicked() {
                        self.game_state = GameState::Menu
                    };
                });

            let ui_data = ui.egui_ctx.end_pass();

            for (id, texture) in &ui_data.textures_delta.set {
                ui.egui_renderer
                    .update_texture(&self.device, &self.queue, *id, texture);
            }
            texture_ui = ui_data.textures_delta.free.clone();
            ui.egui_winit_state
                .handle_platform_output(&win, ui_data.platform_output);
            paint_jobs = ui
                .egui_ctx
                .tessellate(ui_data.shapes, ui_data.pixels_per_point);
            screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point: ui_data.pixels_per_point,
            };
            ui.egui_renderer.update_buffers(
                &self.device,
                &self.queue,
                &mut encoder,
                &paint_jobs,
                &screen_descriptor,
            );
        }
        (paint_jobs, screen_descriptor, texture_ui, Some(fps))
    }
}

impl ApplicationHandler for MyApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = self.set_default_app("My app", event_loop);
        let monitor = window.current_monitor();
        window.set_fullscreen(Some(Fullscreen::Borderless(monitor)));
        let gpu_ctx = pollster::block_on(async {
            let ins = wgpu::Instance::default();
            let surface = ins.create_surface(Arc::clone(&window)).unwrap();
            let adap = ins
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
                .unwrap();
            let (device, queue) = adap
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("MyApp Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                })
                .await
                .unwrap();
            let size = window.inner_size();
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                desired_maximum_frame_latency: 2,
                view_formats: vec![],
            };
            surface.configure(&device, &config);

            let texture_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("tt layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

            let camera_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("camera and model Layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

            let model_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("model_matrices_bind_group_layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: std::num::NonZero::new(64),
                        },
                        count: None,
                    }],
                });

            let mut meshes: Vec<Meshes> = Vec::new();

            let paths = ["./src/door.glb", "./src/ok.glb", "./src/_.glb", "./src/_1.glb"];
            // for path in paths later.. and meshes[i] *for each* file gltf loaded
            for path in paths {
                meshes.extend(load_model(
                    path,
                    &device,
                    &queue,
                    &texture_layout,
                    &model_bind_group_layout,
                ));
            }
            let scene = Scene { meshes };
            // end for return vec

            // load static collision
            let mut rbs = RigidBodySet::new();
            let mut cs = ColliderSet::new();
            let gravity = Vec3::new(0.0, 0.0, 0.0);
            let integration = IntegrationParameters::default();
            let physics_pipeline = PhysicsPipeline::new();
            let island_manager = IslandManager::new();
            let broad_phasebvh = BroadPhaseBvh::new();
            let narrow_phase = NarrowPhase::new();
            let ccd_solver = CCDSolver::new();
            let mut impulse_joint = ImpulseJointSet::new();
            let multi_body_joint = MultibodyJointSet::new();

            let mut doors_handle: Vec<RigidBodyHandle> = Vec::new();

            for mesh in &scene.meshes {
                for primitive in &mesh.primitives {
                    match load_static_collider(
                        &primitive,
                        &mut rbs,
                        &mut cs,
                        &mesh.doors,
                        &mut impulse_joint,
                    ) {
                        Some(door_handle) => doors_handle.push(door_handle),
                        _ => (),
                    };
                }
            }

            // egui
            let egui_ctx = egui::Context::default();
            let egui_winit_state = egui_winit::State::new(
                egui_ctx.clone(),
                egui::ViewportId::ROOT,
                &window,
                None,
                None,
                None,
            );
            let egui_renderer = egui_wgpu::Renderer::new(
                &device,
                config.format,
                egui_wgpu::RendererOptions {
                    msaa_samples: 1,
                    depth_stencil_format: Some(wgpu::TextureFormat::Depth32Float),
                    dithering: true,
                    predictable_texture_filtering: false,
                },
            );
            let ui = UI {
                egui_ctx,
                egui_winit_state,
                egui_renderer,
            };

            let camera = Camera {
                eye: (0.0, 2.25, 5.0).into(),
                target: (0.0, 0.0, 0.0).into(),
                up: Vector3::unit_y(),
                aspect: config.width as f32 / config.height as f32,
                fov: 75.0,
                near: 0.01,
                far: 75.0,
                yaw: -90.0,
                pitch: 0.0,
                is_moving: true,
                is_rotating: true,
            };

            let (char_handle, char_controller) =
                load_player_collision(&camera.eye.into(), &mut rbs, &mut cs);
            // DEFINE iodghvweiugfhwruehfjireughenrfvewfu8g9w4tu3jgrughvfwe 💔
            let collision = Collision {
                rbs,
                cs,
                char_handle,
                integration,
                gravity,
                physics_pipeline,
                island_manager,
                broad_phasebvh,
                narrow_phase,
                ccd_solver,
                impulse_joint,
                multi_body_joint,
                char_controller,
                doors_handle,
            };

            let camera_uniform = UniformCamera {
                uniform: camera.make_camera().into(),
            };

            let camera_planes = Planes::build_plane_from_matrix4(camera_uniform.uniform.into());

            let camera_buffer = device.create_buffer_init(&BufferInitDescriptor {
                label: Some("cmr_buffer"),
                contents: bytemuck::cast_slice(&camera_uniform.uniform),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let shader = device.create_shader_module(wgpu::include_wgsl!("test.wgsl"));

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pipeline layout"),
                bind_group_layouts: &[
                    Some(&camera_bind_group_layout),
                    Some(&texture_layout),
                    Some(&model_bind_group_layout),
                ],
                ..Default::default()
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("pipeline renderer"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 32,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![
                            0 => Float32x3,
                            1 => Float32x3,
                            2 => Float32x2,
                        ],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

            let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("camera bind group"),
                layout: &camera_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }],
            });

            let depth_texture = make_depth_tt(&device, &config);
            let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            let game_state = GameState::Menu;
            let fps_state = PerformanceState {
                mid: true,
                ..Default::default()
            };
            let mut audio = helper::Audio::new();
            audio.load("jog", "./src/audio/jog.mp3");
            audio.load("boom", "./src/audio/boom.mp3");

            WgpuCtx {
                surface,
                device,
                queue,
                config,
                pipeline,
                camera_bind_group,
                camera_buffer,
                camera,
                scene,
                camera_planes,
                depth_view,
                collision,
                ui,
                game_state,
                playing: false,
                mouse_locked: false,
                fps_state,
                audio,
            }
        });

        self.input = InputState::default();
        self.window = Some(window);
        self.gpu_ctx = Some(gpu_ctx);
        self.last_time = Some(Instant::now());
        self.fps = 60.0;
        self.should_draw = true;
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if let Some(gpu) = &mut self.gpu_ctx {
            let state = &mut gpu.ui.egui_winit_state;
            let res = state.on_window_event(&self.window.as_deref().unwrap(), &event);
            let consumed = match gpu.game_state {
                GameState::Play => {
                    self.should_draw = true;
                    false
                }
                _ => {
                    if !gpu.audio.is_playing("boom") {
                        gpu.audio.play_again("boom", "boom", 1.0, 1.0);
                    }
                    res.consumed
                }
            };
            if consumed {
                return;
            } else {
                match event {
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                physical_key: PhysicalKey::Code(KeyCode::Escape),
                                state,
                                ..
                            },
                        ..
                    } => {
                        if gpu.mouse_locked && state.is_pressed() && gpu.playing {
                            if let (Some(win), Some(gpu)) = (&self.window, &mut self.gpu_ctx) {
                                win.set_cursor_visible(true);
                                let _ = win.set_cursor_grab(CursorGrabMode::None);
                                gpu.mouse_locked = false;
                                gpu.game_state = GameState::Menu;
                                gpu.playing = false;
                                self.input = InputState::default();
                                println!("unlock");
                            }
                        }
                    }

                    WindowEvent::Resized(new_size) => {
                        if let Some(gpu) = &mut self.gpu_ctx {
                            gpu.config.width = new_size.width;
                            gpu.config.height = new_size.height;
                            gpu.surface.configure(&gpu.device, &gpu.config);
                            gpu.camera.aspect = new_size.width as f32 / new_size.height as f32;
                            gpu.depth_view = make_depth_tt(&gpu.device, &gpu.config)
                                .create_view(&wgpu::TextureViewDescriptor::default());
                        }
                    }

                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                physical_key: PhysicalKey::Code(key),
                                state,
                                ..
                            },
                        ..
                    } => {
                        let is_pressed = state.is_pressed();
                        if let input = &mut self.input
                            && gpu.playing
                        {
                            match key {
                                KeyCode::KeyW | KeyCode::ArrowUp => input.w = is_pressed,
                                KeyCode::KeyS | KeyCode::ArrowDown => input.s = is_pressed,
                                KeyCode::KeyA | KeyCode::ArrowLeft => input.a = is_pressed,
                                KeyCode::KeyD | KeyCode::ArrowRight => input.d = is_pressed,
                                KeyCode::KeyQ => input.q = is_pressed,
                                KeyCode::KeyE => input.e = is_pressed,
                                KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                                    input.shift = is_pressed
                                }
                                _ => (),
                            }
                        }
                    }

                    WindowEvent::MouseInput {
                        state: winit::event::ElementState::Pressed,
                        button: winit::event::MouseButton::Left,
                        ..
                    } => {
                        self.should_draw = true;
                    }

                    WindowEvent::RedrawRequested => {
                        if let (Some(win), Some(gpu)) = (&mut self.window, &mut self.gpu_ctx) {
                            if let Some((frame, view)) = gpu.get_frame_view() {
                                let mut encoder = gpu.device.create_command_encoder(
                                    &wgpu::CommandEncoderDescriptor {
                                        label: Some("Encoder"),
                                    },
                                );

                                {
                                    match gpu.game_state {
                                        // play
                                        GameState::Play => {
                                            let mut render_pass = helper::game_pass(
                                                &mut encoder,
                                                &view,
                                                &gpu.depth_view,
                                            );
                                            render_pass.set_pipeline(&gpu.pipeline);
                                            render_pass.set_bind_group(
                                                0,
                                                &gpu.camera_bind_group,
                                                &[],
                                            );
                                            for (_, mesh) in gpu.scene.meshes.iter().enumerate() {
                                                render_pass.set_vertex_buffer(
                                                    0,
                                                    mesh.vertex_buffer.slice(..),
                                                );
                                                render_pass.set_index_buffer(
                                                    mesh.index_buffer.slice(..),
                                                    wgpu::IndexFormat::Uint32,
                                                );
                                                let primitives = &mesh.primitives;
                                                for primitive in primitives {
                                                    if gpu.camera_planes.frustum_culling(
                                                        primitive.min,
                                                        primitive.max,
                                                    ) {
                                                        render_pass.set_bind_group(
                                                            1,
                                                            &mesh.textures[primitive.texture_id]
                                                                .texture,
                                                            &[],
                                                        );
                                                        render_pass.set_bind_group(
                                                            2,
                                                            &mesh.bind_group_matrices,
                                                            &[primitive.offset_buffer],
                                                        );
                                                        render_pass.draw_indexed(
                                                            primitive.start
                                                                ..(primitive.start
                                                                    + primitive.count),
                                                            0,
                                                            0..1,
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                        // menu
                                        GameState::Menu => {
                                            let mut render_pass = helper::menu_pass(
                                                &mut encoder,
                                                &view,
                                                &gpu.depth_view,
                                            );
                                            render_pass.set_pipeline(&gpu.pipeline);
                                            render_pass.set_bind_group(
                                                0,
                                                &gpu.camera_bind_group,
                                                &[],
                                            );
                                            let mut static_render_pass =
                                                render_pass.forget_lifetime();
                                            let (paint_jobs, screen_descriptor, texture_ui) =
                                                gpu.update_ui_menu(&win, &mut encoder);
                                            // render
                                            gpu.ui.egui_renderer.render(
                                                &mut static_render_pass,
                                                &paint_jobs,
                                                &screen_descriptor,
                                            );
                                            for id in texture_ui {
                                                gpu.ui.egui_renderer.free_texture(&id);
                                            }
                                        }
                                        // ext
                                        GameState::Exit => {
                                            event_loop.exit();
                                            println!("Exit");
                                        }
                                        // sets
                                        GameState::Settings => {
                                            let mut render_pass = helper::menu_pass(
                                                &mut encoder,
                                                &view,
                                                &gpu.depth_view,
                                            );
                                            render_pass.set_pipeline(&gpu.pipeline);
                                            render_pass.set_bind_group(
                                                0,
                                                &gpu.camera_bind_group,
                                                &[],
                                            );
                                            let mut static_render_pass =
                                                render_pass.forget_lifetime();
                                            let (paint_jobs, screen_descriptor, texture_ui, fps) =
                                                gpu.update_ui_settings(&win, &mut encoder);
                                            match fps {
                                                Some(fps) => {
                                                    if fps != 0.0 {
                                                        self.fps = fps;
                                                    }
                                                }
                                                None => (),
                                            }
                                            // render
                                            gpu.ui.egui_renderer.render(
                                                &mut static_render_pass,
                                                &paint_jobs,
                                                &screen_descriptor,
                                            );
                                            for id in texture_ui {
                                                gpu.ui.egui_renderer.free_texture(&id);
                                            }
                                        }
                                    }
                                } // vitual scope
                                gpu.queue.submit(std::iter::once(encoder.finish()));
                                frame.present();
                            }
                        } else {
                            println!("computer forced to run fast but the data?");
                        }
                    }
                    WindowEvent::CloseRequested => {
                        event_loop.exit();
                        println!("Exiting application...");
                    }
                    _ => (),
                }
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let Some(gpu) = &self.gpu_ctx {
            if !gpu.mouse_locked {
                self.should_draw = true;
                return;
            };
            if let winit::event::DeviceEvent::MouseMotion { delta } = event {
                if let Some(gpu) = &mut self.gpu_ctx {
                    let sensitive: f32 = 0.2;
                    gpu.camera.yaw += (delta.0 as f32) * sensitive;
                    gpu.camera.pitch -= (delta.1 as f32) * sensitive;

                    if gpu.camera.pitch > 89.0 {
                        gpu.camera.pitch = 89.0
                    }
                    if gpu.camera.pitch < -89.0 {
                        gpu.camera.pitch = -89.0
                    }

                    gpu.camera.is_rotating = true;
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let dt = self.update_last_time();
        if let Some(gpu) =
            &mut self.gpu_ctx
        {
            match gpu.game_state {
                GameState::Play => {
                    let input = &self.input;
                    let collision = &mut gpu.collision;
                    let forward = gpu.camera.update_target();
                    let forward_flat = Vector3::new(forward.x, 0.0, forward.z).normalize();
                    let right = forward_flat.cross(gpu.camera.up);
                    let mut velocity = Vector3::new(0.0, 0.0, 0.0);
                    let mut speed = 8.0;
                    if input.shift {
                        speed = 12.0;
                    }

                    if input.w {
                        velocity += forward_flat;
                        gpu.camera.is_moving = true;
                    }
                    if input.s {
                        velocity -= forward_flat;
                        gpu.camera.is_moving = true;
                    }
                    if input.a {
                        velocity -= right;
                        gpu.camera.is_moving = true;
                    }
                    if input.d {
                        velocity += right;
                        gpu.camera.is_moving = true;
                    }
                    if input.q {
                        velocity.y -= 1.0;
                        gpu.camera.is_moving = true;
                    }
                    if input.e {
                        velocity.y += 1.0;
                        gpu.camera.is_moving = true;
                    }

                    collision.physics_pipeline.step(
                        collision.gravity,
                        &collision.integration,
                        &mut collision.island_manager,
                        &mut collision.broad_phasebvh,
                        &mut collision.narrow_phase,
                        &mut collision.rbs,
                        &mut collision.cs,
                        &mut collision.impulse_joint,
                        &mut collision.multi_body_joint,
                        &mut collision.ccd_solver,
                        &(),
                        &(),
                    );
                            helper::ram();
                    if gpu.camera.is_moving || gpu.camera.is_rotating {
                        if !gpu.audio.is_playing("jog") && gpu.camera.is_moving {
                            gpu.audio.play_again("jog", "jog", 1.0, 2.0);
                            helper::ram();
                        } else if !gpu.camera.is_moving {
                            gpu.audio.stop_slowly("jog", 10.0, dt)
                        }

                        for mesh in &gpu.scene.meshes {
                            for primitive in &mesh.primitives {
                                if primitive.is_door.door {
                                    let data: Vec<(Option<Matrix4<f32>>, Option<u32>)> =
                                        collision.check_door(&mesh.doors, &primitive);
                                    let data_unwrap: Vec<(Matrix4<f32>, u32)> =
                                        data.into_iter().filter_map(|(m, o)| m.zip(o)).collect();
                                    for (matrix, offset_buffer) in data_unwrap {
                                        let m: [[f32; 4]; 4] = matrix.into();
                                        gpu.queue.write_buffer(
                                            &mesh.buffer_matrices,
                                            offset_buffer as u64,
                                            bytemuck::cast_slice(&m),
                                        );
                                    }
                                }
                            }
                        }

                        let new_pos =
                            collision.update_check_collision(dt, &velocity.normalize(), speed);
                        gpu.camera.eye = new_pos;

                        gpu.camera.update_target();
                        gpu.camera_planes =
                            Planes::build_plane_from_matrix4(gpu.camera.make_camera());

                        let new_matrix: [[f32; 4]; 4] = gpu.camera.make_camera().into();
                        gpu.queue.write_buffer(
                            &gpu.camera_buffer,
                            0,
                            bytemuck::cast_slice(&new_matrix),
                        );
                        gpu.camera.is_moving = false;
                    }
                }
                _ => {
                    if self.should_draw {
                        let _ = self.update_last_time();
                        self.should_draw = false;
                    } else {()}
                },
            }
        };
    }
}

fn main() {
    let mut app = MyApp::default();
    let event_loop = EventLoop::new().unwrap();
    let _ = event_loop.run_app(&mut app);
}
