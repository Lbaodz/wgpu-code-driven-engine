use cgmath::{Deg, InnerSpace, Matrix4, Point3, Vector3, dot};
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::application::ApplicationHandler;
use winit::event::{KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId, Fullscreen};

#[derive(Default)]
struct Plane {
    normal: [f32;3],
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

    fn frustum_culling(&self, min: [f32;3], max: [f32;3]) -> bool {
        let planes = [&self.left, &self.right, &self.bottom, &self.up, &self.near, &self.far];
        for plane in planes {
            let mut positive_normal: [f32;3] = [0.0;3];
            let normal = plane.normal;
            for i in 0..3 {
                if normal[i] >= 0.0 {
                    positive_normal[i] = max[i];
                } else { positive_normal[i] = min[i]; }
            }
            if dot::<Vector3<f32>>(normal.into(), positive_normal.into()) + plane.d < 0.0 { return false; }
        };
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
    color: [f32; 3],
    normal: [f32; 3],
}

struct Primitive {
    start: u32,
    count: u32,
    min: [f32; 3],
    max: [f32; 3],
    // Some(texture) 
}

struct Meshes {
    vertex_buffer: wgpu::Buffer, 
    index_buffer: wgpu::Buffer, 
    primitves: Vec<Primitive>,
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
    fn set_default_app(&mut self, app_title: &str,event_loop: &ActiveEventLoop) -> Arc<Window> {
        let win_attr = Window::default_attributes().with_title(app_title);
        Arc::new(event_loop.create_window(win_attr).expect("window create failed!"))
    }
}

fn load_model(path: &str, device: &wgpu::Device) -> Vec<Meshes> {
    let (document, buffers, _images) = gltf::import(path).expect("Not found path");
    // return
    let mut objects: Vec<Meshes> = Vec::new();
    // buffer render offset **IMPORTANT: NO SEPARATE BUFFER FOR EACH RETURN**
    let mut all_verticles: Vec<Vertex> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();
    // index range
    let mut pris: Vec<Primitive> = Vec::new();

    for mesh in document.meshes() {

        for primitive in mesh.primitives() {
            let accessor = primitive.get(&gltf::Semantic::Positions).unwrap();
            let (mi, ma) = (accessor.min().unwrap(), accessor.max().unwrap());
            let min: [f32;3] = serde_json::from_value(mi.clone()).expect("cant extract min");
            let max: [f32;3] = serde_json::from_value(ma.clone()).expect("cant extract min");

            let reader = primitive.reader(|b| Some(&buffers[b.index()]));
            let pos: Vec<[f32; 3]> = reader.read_positions().expect("cant get pos").collect();
            let count = pos.len();
            let col: Vec<[f32; 3]> = if let Some(color) = reader.read_colors(0) {
                color.into_rgb_f32().collect()
            } else {
                vec![[1.0, 0.0, 1.0]; count]
            };
            let nor: Vec<[f32; 3]> = reader.read_normals().expect("cant get nor").collect();
            let verticle: Vec<Vertex> = (0..count)
            .map(|i| Vertex { position: pos[i], color: col[i], normal: nor[i] })
            .collect();

            let offset = all_verticles.len() as u32;
            let indices: Vec<u32> = reader.read_indices().unwrap().into_u32()
            .map(|i| i + offset)
            .collect();

            pris.push(
                Primitive { start: offset, count: indices.len() as u32, min, max }
            );

            all_verticles.extend(verticle);
            all_indices.extend(indices);
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
    objects.push(
        Meshes {
            vertex_buffer,
            index_buffer,
            primitves: pris,
        }
    );
    objects
}

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
            let mut meshes: Vec<Meshes> = Vec::new();

            let paths = ["./src/Untitled.gltf"];
            // for path in paths later.. and meshes[i] *for each* file gltf loaded
            for path in paths {
                meshes.extend(load_model(path, &device));
            }
            // end for return vec

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

            let camera_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("cameraLayout"),
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

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pipeline layout"),
                bind_group_layouts: &[Some(&camera_bind_group_layout)],
                ..Default::default()
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 36,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![
                            0 => Float32x3,
                            1 => Float32x3,
                            2 => Float32x3,
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
                        KeyCode::KeyW => input.w = is_pressed,
                        KeyCode::KeyS => input.s = is_pressed,
                        KeyCode::KeyA => input.a = is_pressed,
                        KeyCode::KeyD => input.d = is_pressed,
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
                                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                            store: wgpu::StoreOp::Store,
                                        },
                                    })],
                                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                        view: &gpu.depth_view,
                                        depth_ops: Some(wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(1.0),
                                            store: wgpu::StoreOp::Store,
                                        }),
                                        stencil_ops: None,
                                    }),
                                    ..Default::default()
                                });
                            render_pass.set_pipeline(&gpu.pipeline);
                            render_pass.set_bind_group(0, &gpu.camera_bind_group, &[]);
                            
                            for (_, mesh) in gpu.meshes.iter().enumerate() {
                                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                                let primitives = &mesh.primitves;
                                for primitive in primitives {
                                    if gpu.camera_planes.frustum_culling(primitive.min, primitive.max) {
                                        render_pass.draw_indexed(primitive.start..(primitive.start + primitive.count), 0, 0..1);
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
            self.last_time = Some(Instant::now());
            win.request_redraw();

            if let (Some(gpu), Some(input)) = (&mut self.gpu_ctx, &self.input) {
                let forward = gpu.camera.update_target();
                let forward_flat = Vector3::new(forward.x, 0.0, forward.z).normalize();
                let right = forward_flat.cross(gpu.camera.up);
                let mut velocity = Vector3::new(0.0,0.0,0.0);
                let speed = 0.1;
 
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
                
               if velocity.magnitude2() > 0.0 {
                gpu.camera.eye += velocity.normalize() * speed;
                gpu.camera.update_target();
               }
               gpu.camera_planes = Planes::build_plane_from_matrix4(gpu.camera.make_camera());

                let new_matrix: [[f32; 4]; 4] = gpu.camera.make_camera().into();
                gpu.queue.write_buffer(&gpu.camera_buffer, 0, bytemuck::cast_slice(&new_matrix));
            }
        };
    }
}

fn main() {
    let mut app = MyApp::default();
    let event_loop = EventLoop::new().unwrap();
    let _ = event_loop.run_app(&mut app);
}
