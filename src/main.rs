use cgmath::{Deg, InnerSpace, Matrix4, Point3, Vector3, Vector4, Quaternion, dot};
use rapier3d::dynamics::RigidBodyHandle;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::application::ApplicationHandler;
use winit::event::{KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};
use rapier3d::{control::{KinematicCharacterController, CharacterAutostep, CharacterLength}, pipeline::{QueryFilter, PhysicsPipeline},dynamics::{RigidBodySet, RigidBodyBuilder, ImpulseJointSet, MultibodyJointSet, CCDSolver, IslandManager, IntegrationParameters}, geometry::{Group, ColliderBuilder, ColliderSet, InteractionGroups, InteractionTestMode, BroadPhaseBvh, NarrowPhase}, math::Vec3};

const DYN: Group = Group::GROUP_1;
const STA: Group = Group::GROUP_2;

#[derive(Default)]
struct Plane {
    normal: [f32; 3],
    d: f32,
}

struct Planes {
    left: Plane,
    right: Plane,
    bottom: Plane,
    up: Plane,
    near: Plane,
    far: Plane,
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
            left: make_plane(m.x.w + m.x.x, m.y.w + m.y.x, m.z.w + m.z.x, m.w.w + m.w.x),
            right: make_plane(m.x.w - m.x.x, m.y.w - m.y.x, m.z.w - m.z.x, m.w.w - m.w.x),
            bottom: make_plane(m.x.w + m.x.y, m.y.w + m.y.y, m.z.w + m.z.y, m.w.w + m.w.y),
            up: make_plane(m.x.w - m.x.y, m.y.w - m.y.y, m.z.w - m.z.y, m.w.w - m.w.y),
            near: make_plane(m.x.w + m.x.z, m.y.w + m.y.z, m.z.w + m.z.z, m.w.w + m.w.z),
            far: make_plane(m.x.w - m.x.z, m.y.w - m.y.z, m.z.w - m.z.z, m.w.w - m.w.z),
        }
    }

    fn frustum_culling(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        let planes = [
            &self.left,
            &self.right,
            &self.bottom,
            &self.up,
            &self.near,
            &self.far,
        ];
        for plane in planes {
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
#[derive(Debug)]
struct Primitive {
    start: u32,
    count: u32,
    min: [f32; 3],
    max: [f32; 3],
    center: Vector3<f32>,
    extent: Vector3<f32>,
    texture_id: usize,
    mmbg: wgpu::BindGroup,
}

struct Texture {
    texture: wgpu::BindGroup,
}

struct Meshes {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    primitves: Vec<Primitive>,
    textures: Vec<Texture>,
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
    v_fall: f32,
}

impl Collision {
    fn update_check_collision(&mut self, dt: f32, desire_movement: &Vector3<f32>, speed: f32) -> Point3<f32> {
        let char_data = &self.rbs[self.char_handle];
        let char_collider_handle = char_data.colliders()[0];
        let (character_shape, character_pos) = (&self.cs[char_collider_handle].shape(), char_data.position());
        let query_pipeline = self.broad_phasebvh.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
        &self.rbs, &self.cs, QueryFilter::default().exclude_collider(char_collider_handle));

        let movement_result = self.char_controller.move_shape(dt,
            &query_pipeline, *character_shape, 
            character_pos, 
            Vec3::new(desire_movement.x, desire_movement.y, desire_movement.z) * speed * dt,
            |_|()
        );

        self.physics_pipeline.step(self.gravity, &self.integration,
        &mut self.island_manager, &mut self.broad_phasebvh,
        &mut self.narrow_phase, &mut self.rbs, &mut self.cs,
        &mut self.impulse_joint, &mut self.multi_body_joint, &mut self.ccd_solver,
        &(), &());

        /*  TWO SIDES INTERACT ONLY
        self.char_controller.solve_character_collision_impulses(
            dt,
            &mut query_pipeline_mut, 
            *character_shape, 4.0, 
            movement_result.collision,
        );
        */
        if movement_result.grounded {
            self.v_fall = 0.0;
        }
        let mut rb = &mut self.rbs[self.char_handle];
        let new_pos = rb.position().translation + movement_result.translation;
        rb.set_next_kinematic_translation(new_pos);
        Point3::new(new_pos.x, new_pos.y, new_pos.z)
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
    meshes: Vec<Meshes>,
    depth_view: wgpu::TextureView,
    collision: Collision,
}

#[derive(Default)]
struct MyApp {
    window: Option<Arc<Window>>,
    gpu_ctx: Option<WgpuCtx>,
    last_time: Option<Instant>,
    input: Option<InputState>,
    mouse_locked: bool,
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
}

fn load_static_collider(primitive: &Primitive, rbs: &mut RigidBodySet, cs: &mut ColliderSet) {
    let static_group = InteractionGroups::new(STA, DYN, InteractionTestMode::Or);
    let extent = primitive.extent;
    let center = primitive.center;
    let rb = RigidBodyBuilder::fixed()
    .translation(Vec3::new(center.x, center.y, center.z)).build();

    let collider = ColliderBuilder::cuboid(extent.x, extent.y, extent.z)
    .collision_groups(static_group)
    .build();
    let rb_handle = rbs.insert(rb);
    cs.insert_with_parent(collider, rb_handle, rbs);
}

fn load_player_collision(pos: &[f32; 3], rbs: &mut RigidBodySet, cs: &mut ColliderSet) -> (RigidBodyHandle, KinematicCharacterController) {
    let dyn_group = InteractionGroups::new(DYN, STA, InteractionTestMode::Or);
    let rb = RigidBodyBuilder::kinematic_position_based()
    .translation(Vec3::new(pos[0], pos[1], pos[2])).build();
    let rb_handle = rbs.insert(rb);

    let collider = ColliderBuilder::capsule_y(2.0, 0.3)
    .collision_groups(dyn_group)
    .build();
    cs.insert_with_parent(collider, rb_handle, rbs);
    let mut char_controller = KinematicCharacterController::default();
    char_controller.autostep = Some(CharacterAutostep { 
        max_height: CharacterLength::Absolute(0.5),
        min_width: CharacterLength::Absolute(0.2),
        include_dynamic_bodies: true,
    });
    char_controller.snap_to_ground = Some(CharacterLength::Relative(10.0));

    (rb_handle, char_controller)
}

fn convert_aabb_from_matrix(min: Vector3<f32>, max: Vector3<f32>, matrix: Matrix4<f32>) -> ([f32;3], [f32;3], Vector3<f32>, Vector3<f32>) {
    let old_extent = (max - min) * 0.5;
    let old_center = (min + max) * 0.5;
    let center4 = matrix * Vector4::new(old_center.x, old_center.y, old_center.z, 1.0);
    let center = Vector3::new(center4.x, center4.y, center4.z);

    let mut extent = Vector3::new(0.0,0.0,0.0);
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
    for scene in document.scenes() {
        for node in scene.nodes() {
            // matrix model
            let model_matrix: [[f32; 4]; 4]  = match node.transform() {
                gltf::scene::Transform::Matrix { matrix } => matrix,
                gltf::scene::Transform::Decomposed { translation, rotation, scale } => {
                    let t: Matrix4<f32> = Matrix4::from_translation(Vector3::new(translation[0], translation[1], translation[2]));
                    let r: Matrix4<f32> = Matrix4::from(Quaternion::new(rotation[3], rotation[0], rotation[1], rotation[2]));
                    let s: Matrix4<f32> = Matrix4::from_nonuniform_scale(scale[0], scale[1], scale[2]);
                    (t * r * s).into()
                }
            };

            let model_matrix_buffer = device.create_buffer_init(&BufferInitDescriptor {
                label: Some("model_matrix_buffer"),
                contents: bytemuck::cast_slice(&model_matrix),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let model_matrix_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:  Some("model matrix"), 
                layout: &model_matrix_layout, 
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model_matrix_buffer.as_entire_binding(),
                }],
            });

            if let Some(mesh) = node.mesh() {
                for primitive in mesh.primitives() {
                    // min max
                    let accessor = primitive.get(&gltf::Semantic::Positions).unwrap();
                    let (mi, ma) = (accessor.min().unwrap(), accessor.max().unwrap());
                    let min_raw: [f32; 3] =
                        serde_json::from_value(mi.clone()).expect("cant extract min");
                    let max_raw: [f32; 3] =
                        serde_json::from_value(ma.clone()).expect("cant extract min");
                    let (min, max, center, extent) = convert_aabb_from_matrix(min_raw.into(), max_raw.into(), model_matrix.into());

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
                            address_mode_u: wgpu::AddressMode::ClampToEdge,
                            address_mode_v: wgpu::AddressMode::ClampToEdge,
                            address_mode_w: wgpu::AddressMode::ClampToEdge,
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

                    let mmbg = model_matrix_bind_group.clone();
                    pris.push(Primitive {
                        start: start,
                        count: indices.len() as u32,
                        min,
                        max,
                        center,
                        extent,
                        texture_id,
                        mmbg,
                    });

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
    objects.push(Meshes {
        vertex_buffer,
        index_buffer,
        primitves: pris,
        textures,
    });
    objects
}

// break line <------------------------------------------------------------------------------------>
impl WgpuCtx {
    pub fn get_frame_view(&self) -> Option<(wgpu::SurfaceTexture, wgpu::TextureView)> {
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

            let mut meshes: Vec<Meshes> = Vec::new();

            let paths = ["./src/ok.glb"];
            // for path in paths later.. and meshes[i] *for each* file gltf loaded
            for path in paths {
                meshes.extend(load_model(path, &device, &queue, &texture_layout, &camera_bind_group_layout));
            }
            // end for return vec

            // load static collision
            let mut rbs = RigidBodySet::new();
            let mut cs = ColliderSet::new();
            let gravity = Vec3::new(0.0, 0.0, 0.0);
            let integration = IntegrationParameters::default();
            let mut physics_pipeline = PhysicsPipeline::new();
            let mut island_manager = IslandManager::new();
            let mut broad_phasebvh = BroadPhaseBvh::new();
            let mut narrow_phase = NarrowPhase::new();
            let mut ccd_solver = CCDSolver::new();
            let mut impulse_joint = ImpulseJointSet::new();
            let mut multi_body_joint = MultibodyJointSet::new();
            let mut query_pipeline_mut = broad_phasebvh.as_query_pipeline_mut(
                narrow_phase.query_dispatcher(),
                &mut rbs,
                &mut cs,
                QueryFilter::default()
            );
            for mesh in &meshes {
                for primitive in &mesh.primitves {
                    load_static_collider(&primitive, &mut rbs, &mut cs);
                }
            }

            let camera = Camera {
                eye: (0.0, 2.0, 5.0).into(),
                target: (0.0, 0.0, 0.0).into(),
                up: Vector3::unit_y(),
                aspect: config.width as f32 / config.height as f32,
                fov: 75.0,
                near: 0.1,
                far: 100.0,
                yaw: -90.0,
                pitch: 0.0,
            };

            let (char_handle, char_controller) = load_player_collision(&camera.eye.into(), &mut rbs, &mut cs);
            // DEFINE iodghvweiugfhwruehfjireughenrfvewfu8g9w4tu3jgrughvfwe 💔
            let collision = Collision {
                rbs, cs, char_handle, integration, gravity,
                physics_pipeline, island_manager, broad_phasebvh,
                narrow_phase, ccd_solver, impulse_joint, multi_body_joint,
                char_controller, v_fall: 0.0,
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
                bind_group_layouts: &[Some(&camera_bind_group_layout), Some(&texture_layout), Some(&camera_bind_group_layout)],
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

            let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("depth tt"),
                format: wgpu::TextureFormat::Depth32Float,
                dimension: wgpu::TextureDimension::D2,
                mip_level_count: 1,
                sample_count: 1,
                size: wgpu::Extent3d {
                    width: config.width,
                    height: config.height,
                    depth_or_array_layers: 1,
                },
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[wgpu::TextureFormat::Depth32Float],
            });
            let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            WgpuCtx {
                surface,
                device,
                queue,
                config,
                pipeline,
                camera_bind_group,
                camera_buffer,
                camera,
                meshes,
                camera_planes,
                depth_view,
                collision,
            }
        });

        self.input = Some(InputState::default());
        self.window = Some(window);
        self.gpu_ctx = Some(gpu_ctx);
        self.last_time = Some(Instant::now());
        self.mouse_locked = false;
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
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
                if self.mouse_locked && state.is_pressed() {
                    if let Some(win) = &self.window {
                        win.set_cursor_visible(true);
                        let _ = win.set_cursor_grab(CursorGrabMode::None);
                        self.mouse_locked = false;
                    }
                }
            }

            WindowEvent::MouseInput {
                state: winit::event::ElementState::Pressed,
                button: winit::event::MouseButton::Left,
                ..
            } => {
                if !self.mouse_locked {
                    if let Some(win) = &self.window {
                        win.set_cursor_visible(false);
                        let _ = win.set_cursor_grab(CursorGrabMode::Locked);
                        self.mouse_locked = true;
                    }
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
                if let Some(input) = &mut self.input {
                    match key {
                        KeyCode::KeyW | KeyCode::ArrowUp => input.w = is_pressed,
                        KeyCode::KeyS | KeyCode::ArrowDown => input.s = is_pressed,
                        KeyCode::KeyA | KeyCode::ArrowLeft => input.a = is_pressed,
                        KeyCode::KeyD | KeyCode::ArrowRight => input.d = is_pressed,
                        KeyCode::KeyQ => input.q = is_pressed,
                        KeyCode::KeyE => input.e = is_pressed,
                        _ => (),
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                if let (Some(win), Some(gpu)) = (&self.window, &self.gpu_ctx) {
                    if let Some((frame, view)) = gpu.get_frame_view() {
                        let mut encoder =
                            gpu.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("Encoder"),
                                });

                        {
                            let mut render_pass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("render pass"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &view,
                                        depth_slice: None,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                            store: wgpu::StoreOp::Store,
                                        },
                                    })],
                                    depth_stencil_attachment: Some(
                                        wgpu::RenderPassDepthStencilAttachment {
                                            view: &gpu.depth_view,
                                            depth_ops: Some(wgpu::Operations {
                                                load: wgpu::LoadOp::Clear(1.0),
                                                store: wgpu::StoreOp::Store,
                                            }),
                                            stencil_ops: None,
                                        },
                                    ),
                                    ..Default::default()
                                });
                            render_pass.set_pipeline(&gpu.pipeline);
                            render_pass.set_bind_group(0, &gpu.camera_bind_group, &[]);

                            for (_, mesh) in gpu.meshes.iter().enumerate() {
                                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                render_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                let primitives = &mesh.primitves;
                                for primitive in primitives {
                                    if gpu
                                        .camera_planes
                                        .frustum_culling(primitive.min, primitive.max)
                                    {
                                        render_pass.set_bind_group(
                                            1,
                                            &mesh.textures[primitive.texture_id].texture,
                                            &[],
                                        );
                                        render_pass.set_bind_group(2, &primitive.mmbg, &[]);
                                        render_pass.draw_indexed(
                                            primitive.start..(primitive.start + primitive.count),
                                            0,
                                            0..1,
                                        );
                                    }
                                }
                            }
                        }
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

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if self.mouse_locked {
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

                    gpu.camera.update_target();
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let (Some(win), Some(last_time)) = (&self.window, &self.last_time) {
            let desire_time = Duration::from_secs_f64(1.0 / 90.0);
            let time_eslaped = last_time.elapsed();
            if time_eslaped < desire_time {
                let sleep_time = desire_time - time_eslaped;
                std::thread::sleep(sleep_time);
            };
            let now = Instant::now();
            let dt = now.duration_since(*last_time).as_secs_f32();
            self.last_time = Some(now);
            win.request_redraw();

            if let (Some(gpu), Some(input)) = (&mut self.gpu_ctx, &self.input) {
                let collision = &mut gpu.collision;
                let forward = gpu.camera.update_target();
                let forward_flat = Vector3::new(forward.x, 0.0, forward.z).normalize();
                let right = forward_flat.cross(gpu.camera.up);
                let mut velocity = Vector3::new(0.0, 0.0, 0.0);
                let speed = 4.0;
                
                if input.w {
                    velocity += forward_flat
                }
                if input.s {
                    velocity -= forward_flat
                }
                if input.a {
                    velocity -= right
                }
                if input.d {
                    velocity += right
                }
                if input.q {
                    velocity.y -= 1.0
                }
                if input.e {
                    velocity.y += 1.0
                }

                let new_pos = collision.update_check_collision(dt, &velocity.normalize(), speed);
                gpu.camera.eye = new_pos;
                gpu.camera.update_target();
                gpu.camera_planes = Planes::build_plane_from_matrix4(gpu.camera.make_camera());

                let new_matrix: [[f32; 4]; 4] = gpu.camera.make_camera().into();
                gpu.queue
                    .write_buffer(&gpu.camera_buffer, 0, bytemuck::cast_slice(&new_matrix));
            }
        };
    }
}

fn main() {
    let mut app = MyApp::default();
    let event_loop = EventLoop::new().unwrap();
    let _ = event_loop.run_app(&mut app);
}