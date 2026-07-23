mod wgpu_ctx;
use wgpu_ctx::game_manager::{GameState, InputState, GameLevel, PerformanceState, FileManager};
use cgmath::{InnerSpace, Matrix4, Vector3};
use rapier3d::{
    control::KinematicCharacterController,
    dynamics::{
        CCDSolver, ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyJointSet,
        RigidBodySet, ImpulseJointHandle, RigidBodyHandle
    },
    geometry::{
        BroadPhaseBvh, ColliderSet,
        NarrowPhase,
    },
    math::Vec3,
    pipeline::PhysicsPipeline,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;
use std::vec;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::application::ApplicationHandler;
use winit::event::{KeyEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};
use wgpu_ctx::{
    ui::UI, scene::{
        audio,
        camera::{Camera, Planes, UniformCamera},
        meshes::collision::Collision, ResultSent, Scene
    }
};
use wgpu_ctx::wgpu_helper;
use wgpu_ctx::scene::scene_helper;

#[derive(Default)]
pub struct MyApp {
    window: Option<Arc<Window>>,
    gpu_ctx: Option<wgpu_ctx::WgpuCtx>,
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
                thread::sleep(sleep_time);
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

impl ApplicationHandler for MyApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = self.set_default_app("NoGame", event_loop);
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
                    label: Some("Device vs queue"),
                    required_features: wgpu::Features::TEXTURE_COMPRESSION_BC,
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
                format: wgpu::TextureFormat::Bgra8Unorm,
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

            let rbs = RigidBodySet::new();
            let cs = ColliderSet::new();
            let gravity = Vec3::new(0.0, 0.0, 0.0);
            let integration = IntegrationParameters::default();
            let physics_pipeline = PhysicsPipeline::new();
            let island_manager = IslandManager::new();
            let broad_phasebvh = BroadPhaseBvh::new();
            let narrow_phase = NarrowPhase::new();
            let ccd_solver = CCDSolver::new();
            let impulse_joint = ImpulseJointSet::new();
            let multi_body_joint = MultibodyJointSet::new();

            let doors_handle: Vec<RigidBodyHandle> = Vec::new();
            let joints_handle: Vec<ImpulseJointHandle> = Vec::new();

            let camera = Camera {
                eye: (0.0, 2.7, 5.0).into(),
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
            let loader_progress: Arc<AtomicU32> = Arc::new(AtomicU32::new(0));

            let (_, rr) = channel::<ResultSent>();

            let model_paths: Vec<String> = vec![
                "catdoor",
                /*"_",
                "_1",
                "do",
                "do1",
                "ok",
                "door", */
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
            let md_len = model_paths.len();
            let audio_paths: Vec<String> = vec!["jog.mp3", "boom.mp3"]
                .into_iter()
                .map(|s| s.to_string())
                .collect();

            let mut file_manager = HashMap::new();
            file_manager.insert(
                String::from("base scene"),
                FileManager {
                    model_paths,
                    audio_paths,
                },
            );

            let scene = Scene {
                meshes: Vec::with_capacity(md_len),
                rr,
            };

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

            // DEFINE iodghvweiugfhwruehfjireughenrfvewfu8g9w4tu3jgrughvfwe 💔
            let collision = Collision {
                rbs,
                cs,
                char_handle: RigidBodyHandle::default(),
                integration,
                gravity,
                physics_pipeline,
                island_manager,
                broad_phasebvh,
                narrow_phase,
                ccd_solver,
                impulse_joint,
                multi_body_joint,
                char_controller: KinematicCharacterController::default(),
                doors_handle,
                joints_handle,
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

            let depth_texture = wgpu_helper::make_depth_tt(&device, &config);
            let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            let game_state = GameState::Loading; // default
            let fps_state = PerformanceState {
                mid: true,
                ..Default::default()
            };

            let audio = audio::Audio::new();

            wgpu_ctx::WgpuCtx {
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
                texture_layout,
                mbg_layout: model_bind_group_layout,
                depth_view,
                collision,
                ui,
                game_state,
                playing: false,
                mouse_locked: false,
                fps_state,
                audio,
                file_manager,
                should_load: Arc::new(AtomicBool::new(true)),
                loader_progress,
                game_level: GameLevel::Base, // default
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
                GameState::Loading => res.consumed,
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
                            }
                        }
                    }

                    WindowEvent::Resized(new_size) => {
                        if let Some(gpu) = &mut self.gpu_ctx {
                            gpu.config.width = new_size.width;
                            gpu.config.height = new_size.height;
                            gpu.surface.configure(&gpu.device, &gpu.config);
                            gpu.camera.aspect = new_size.width as f32 / new_size.height as f32;
                            gpu.depth_view = wgpu_helper::make_depth_tt(&gpu.device, &gpu.config)
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
                        gpu.audio.stop("jog");
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
                                            let mut render_pass = wgpu_helper::game_pass(
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
                                                    ) || true
                                                    {
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
                                            let render_pass = wgpu_helper::menu_pass(
                                                &mut encoder,
                                                &view,
                                                &gpu.depth_view,
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
                                            let render_pass = wgpu_helper::menu_pass(
                                                &mut encoder,
                                                &view,
                                                &gpu.depth_view,
                                            );
                                            let mut static_render_pass =
                                                render_pass.forget_lifetime();
                                            let (paint_jobs, screen_descriptor, texture_ui, fps) =
                                                gpu.update_ui_settings(&win, &mut encoder);
                                            if let Some(fps) = fps {
                                                if fps != 0.0 {
                                                    self.fps = fps
                                                }
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

                                        GameState::Loading => {
                                            let render_pass = wgpu_helper::menu_pass(
                                                &mut encoder,
                                                &view,
                                                &gpu.depth_view,
                                            );
                                            let mut static_render_pass =
                                                render_pass.forget_lifetime();
                                            let (paint_jobs, screen_descriptor, texture_ui) =
                                                gpu.update_ui_loading(&win, &mut encoder);
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
        if let Some(gpu) = &mut self.gpu_ctx {
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

                    if collision.need_update_door() {
                        collision.update_door(&velocity);
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
                    //helper::ram();
                    if !gpu.audio.is_playing("jog") && gpu.camera.is_moving {
                        gpu.audio.play_again("jog", "jog", 1.0, 2.0);
                    } else if !gpu.camera.is_moving {
                        gpu.audio.stop_slowly("jog", 10.0, dt);
                    }

                    if gpu.camera.is_moving || gpu.camera.is_rotating {
                        if gpu.camera.is_rotating && !gpu.camera.is_moving {
                            gpu.camera.update_target();
                            gpu.camera_planes =
                                Planes::build_plane_from_matrix4(gpu.camera.make_camera());

                            let new_matrix: [[f32; 4]; 4] = gpu.camera.make_camera().into();
                            gpu.queue.write_buffer(
                                &gpu.camera_buffer,
                                0,
                                bytemuck::cast_slice(&new_matrix),
                            );
                            gpu.camera.is_rotating = false;
                        } else {
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
                            gpu.camera.is_rotating = false;
                        }
                    }
                }
                _ => {
                    if gpu.should_load.load(std::sync::atomic::Ordering::Relaxed) {
                        match gpu.game_level {
                            GameLevel::Base => {
                                gpu.load_all(String::from("base scene"));
                            }
                            _ => (),
                        }
                    }
                    if let Ok(result) = gpu.scene.rr.try_recv() {
                        let collision = &mut gpu.collision;
                        gpu.game_state = GameState::Menu;
                        gpu.scene.meshes = result.meshes;
                        collision.rbs = result.rbs;
                        collision.cs = result.cs;
                        collision.impulse_joint = result.impulse_joint;
                        collision.doors_handle = result.doors_handle;
                        collision.joints_handle = result.joints_handle;
                        collision.char_controller = result.char_controller;
                        collision.char_handle = result.char_handle;
                        gpu.audio = result.audio;
                    }
                    if self.should_draw {
                        scene_helper::ram();
                        let _ = self.update_last_time();
                        self.should_draw = false;
                    } else {
                        ()
                    }
                }
            }
        };
    }
}