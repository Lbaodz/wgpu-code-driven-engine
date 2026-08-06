mod wgpu_ctx;
use glam::Vec3;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::mpsc::channel;
use std::thread;
use std::time::{Duration, Instant};
use std::vec;
use wgpu::Features;
use wgpu_ctx::game_manager::{FileManager, GameLevel, GameState, InputState, PerformanceState};
use wgpu_ctx::scene::camera::Light;
use wgpu_ctx::scene::scene_helper;
use wgpu_ctx::wgpu_helper;
use wgpu_ctx::{
    scene::{ResultSent, Scene, audio, camera::Planes, meshes::collision::Collision},
    ui::UI,
};
use winit::application::ApplicationHandler;
use winit::event::{KeyEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};

use crate::app::wgpu_ctx::WgpuCtx;

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
        scene_helper::ram("very scratch loader");
        let c = Instant::now();
        let window = self.set_default_app("NoGame", event_loop);
        let monitor = window.current_monitor();
        window.set_fullscreen(Some(Fullscreen::Borderless(monitor)));
        let gpu_ctx = pollster::block_on(async {
            let ins = wgpu::Instance::default();
            let surface = ins
                .create_surface(Arc::clone(&window))
                .expect("can create surface");
            let adap = ins
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
                .expect("cant create adap");
            let (device, queue) = adap
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("Device vs queue"),
                    required_features: Features::TEXTURE_COMPRESSION_BC
                        | Features::VERTEX_WRITABLE_STORAGE,
                    required_limits: wgpu::Limits::defaults(),
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                })
                .await
                .expect("cant create device");
            let size = window.inner_size();
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: wgpu::TextureFormat::Bgra8Unorm,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                desired_maximum_frame_latency: 2,
                view_formats: vec![],
            };
            surface.configure(&device, &config);
            println!("loaded window/wgpu general: {}", c.elapsed().as_millis());

            let loader_progress: Arc<AtomicU32> = Arc::new(AtomicU32::new(0));

            let (_, rr) = channel::<ResultSent>();

            let model_paths: Vec<String> = vec![
                "catdoor_edited",
                //"catdoor_fixed",
                //"do",
                //"do1",
                //"ok",
                //"door",
            ]
            .into_iter()
            .map(|s: &str| s.to_string())
            .collect();
            let transparency_paths: Vec<String> = vec![
                //"catdoor_edited",
                //"catdoor_fixed",
                //"do",
                //"do1",
                //"ok",
                "door",
            ]
            .into_iter()
            .map(|s: &str| s.to_string())
            .collect();
            let md_len = model_paths.len();
            let tr_len = transparency_paths.len();
            let audio_paths: Vec<String> =
                vec!["jog.mp3"].into_iter().map(|s| s.to_string()).collect();

            let mut file_manager = HashMap::new();
            file_manager.insert(
                String::from("base scene"),
                FileManager {
                    model_paths,
                    transparency_paths,
                    audio_paths,
                },
            );

            let scene = Scene {
                meshes: Vec::with_capacity(md_len),
                transparency_meshes: Vec::with_capacity(tr_len),
                lights: vec![Light::default()],
                light_first_loaded: false,
                rr,
                loaded: false,
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

            let collision = Collision::default();

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
                render_ctx: None,
                light_ctx: None,
                scene,
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
        println!("resumed finish: {}ms", c.elapsed().as_millis());
        scene_helper::ram("resumed");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if let Some(gpu) = &mut self.gpu_ctx {
            let state = &mut gpu.ui.egui_winit_state;
            let res = state.on_window_event(
                &self.window.as_deref().expect("cant take event res window"),
                &event,
            );
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
                            if gpu.should_load.load(std::sync::atomic::Ordering::Relaxed) == true {
                                return;
                            };
                            let Some(render_ctx) = &mut gpu.render_ctx else {
                                panic!("cant get render ctx");
                            };
                            gpu.config.width = new_size.width;
                            gpu.config.height = new_size.height;
                            gpu.surface.configure(&gpu.device, &gpu.config);
                            render_ctx.camera.aspect =
                                new_size.width as f32 / new_size.height as f32;
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
                                            let Some(light_ctx) = &gpu.light_ctx else {
                                                panic!("cant get render ctx");
                                            };
                                            let Some(render_ctx) = &gpu.render_ctx else {
                                                panic!("cant get render ctx");
                                            };
                                            let all_pipeline = &render_ctx.pipeline;
                                            // z light
                                            if gpu.collision.need_update_door()
                                                || !gpu.scene.light_first_loaded
                                            {
                                                gpu.scene.draw_light(&mut encoder, &light_ctx);
                                            }
                                            // early z
                                            {
                                                let mut z_pass = wgpu_helper::early_z_pass(
                                                    &mut encoder,
                                                    &gpu.depth_view,
                                                );
                                                z_pass.set_pipeline(
                                                    &all_pipeline.early_depth_pipeline,
                                                );
                                                z_pass.set_bind_group(
                                                    0,
                                                    &render_ctx.camera_bind_group,
                                                    &[],
                                                );
                                                gpu.draw_meshes(
                                                    &gpu.scene.meshes,
                                                    &mut z_pass,
                                                    false,
                                                );
                                            }
                                            // compute pass
                                            {
                                                let mut c_pass =
                                                    wgpu_helper::create_compute_pass(&mut encoder);
                                                c_pass.set_pipeline(&all_pipeline.compute_pipeline);
                                                c_pass.set_bind_group(
                                                    0,
                                                    &render_ctx.camera_bind_group,
                                                    &[],
                                                );
                                                c_pass.set_bind_group(
                                                    1,
                                                    &light_ctx.compute_lights_bg,
                                                    &[],
                                                );
                                                c_pass.dispatch_workgroups(16, 16, 1);
                                            }
                                            // opaque/transparency pass
                                            let mut render_pass = wgpu_helper::game_pass(
                                                &mut encoder,
                                                &view,
                                                &gpu.depth_view,
                                            );
                                            // opaque draw
                                            render_pass.set_pipeline(&all_pipeline.render_pipeline);
                                            WgpuCtx::bind_basic_bg(&mut render_pass, &render_ctx, &light_ctx);
                                            gpu.draw_meshes(
                                                &gpu.scene.meshes,
                                                &mut render_pass,
                                                true,
                                            );
                                            // transparency draw
                                            render_pass.set_pipeline(&all_pipeline.transparency_pipeline);
                                            gpu.draw_meshes(
                                                &gpu.scene.transparency_meshes,
                                                &mut render_pass,
                                                true,
                                            );
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
                                            scene_helper::ram("frame record");
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
                                gpu.device
                                    .poll(wgpu::PollType::Wait {
                                        submission_index: None,
                                        timeout: None,
                                    })
                                    .expect("no dv");
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
                    let Some(render_ctx) = &mut gpu.render_ctx else {
                        panic!("cant get render ctx");
                    };
                    let sensitive: f32 = 0.2;
                    render_ctx.camera.yaw += (delta.0 as f32) * sensitive;
                    render_ctx.camera.pitch -= (delta.1 as f32) * sensitive;

                    if render_ctx.camera.pitch > 89.0 {
                        render_ctx.camera.pitch = 89.0
                    }
                    if render_ctx.camera.pitch < -89.0 {
                        render_ctx.camera.pitch = -89.0
                    }

                    render_ctx.camera.is_rotating = true;
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let dt = self.update_last_time();
        if let Some(gpu) = &mut self.gpu_ctx {
            match gpu.game_state {
                GameState::Play => {
                    let Some(render_ctx) = &mut gpu.render_ctx else {
                        panic!("cant get render ctx");
                    };
                    let input = &self.input;
                    let collision = &mut gpu.collision;
                    let forward = render_ctx.camera.update_target();
                    let forward_flat = Vec3::new(forward.x, 0.0, forward.z).normalize();
                    let right = forward_flat.cross(render_ctx.camera.up);
                    let mut velocity = Vec3::new(0.0, 0.0, 0.0);
                    let mut speed = 8.0;
                    if input.shift {
                        speed = 12.0;
                    }

                    if input.w {
                        velocity += forward_flat;
                        render_ctx.camera.is_moving = true;
                    }
                    if input.s {
                        velocity -= forward_flat;
                        render_ctx.camera.is_moving = true;
                    }
                    if input.a {
                        velocity -= right;
                        render_ctx.camera.is_moving = true;
                    }
                    if input.d {
                        velocity += right;
                        render_ctx.camera.is_moving = true;
                    }
                    if input.q {
                        velocity.y -= 1.0;
                        render_ctx.camera.is_moving = true;
                    }
                    if input.e {
                        velocity.y += 1.0;
                        render_ctx.camera.is_moving = true;
                    }

                    // let make primitve for DOOR CACHE
                    if collision.need_update_door() {
                        for mesh in &mut gpu.scene.meshes {
                            for primitive in &mut mesh.primitives {
                                if primitive.is_door.door {
                                    let Some(id) = primitive.is_door.id else {
                                        panic!("no id door");
                                    };
                                    collision.update_door(&velocity, id);
                                    let (min, max) = collision.new_door_min_max(id as u128);
                                    primitive.min = min;
                                    primitive.max = max;
                                    collision.update_matrix_door(
                                        id,
                                        &mesh.doors,
                                        &primitive,
                                        &gpu.queue,
                                        &mesh.buffer_matrices,
                                    );
                                }
                            }
                        }
                    }

                    collision.update_physics(dt);

                    //helper::ram();
                    if !gpu.audio.is_playing("jog") && render_ctx.camera.is_moving {
                        gpu.audio.play_again("jog", "jog", 1.0, 2.0);
                    } else if !render_ctx.camera.is_moving {
                        gpu.audio.stop_slowly("jog", 10.0, dt);
                    }

                    if render_ctx.camera.is_moving || render_ctx.camera.is_rotating {
                        if render_ctx.camera.is_rotating && !render_ctx.camera.is_moving {
                            render_ctx.camera.update_target();
                            render_ctx
                                .camera
                                .update_planes(Planes::build_plane_from_matrix4(
                                    render_ctx.camera.make_camera(),
                                ));

                            let new_matrix = render_ctx.camera.make_camera();
                            gpu.queue.write_buffer(
                                &render_ctx.camera_buffer,
                                0,
                                bytemuck::bytes_of(&new_matrix),
                            );
                            render_ctx.camera.is_rotating = false;
                        } else {
                            let new_pos =
                                collision.update_check_collision(dt, &velocity.normalize(), speed);
                            render_ctx.camera.eye = new_pos;

                            render_ctx.camera.update_target();
                            render_ctx
                                .camera
                                .update_planes(Planes::build_plane_from_matrix4(
                                    render_ctx.camera.make_camera(),
                                ));

                            let new_matrix = render_ctx.camera.make_camera();
                            gpu.queue.write_buffer(
                                &render_ctx.camera_buffer,
                                0,
                                bytemuck::bytes_of(&new_matrix),
                            );
                            render_ctx.camera.is_moving = false;
                            render_ctx.camera.is_rotating = false;
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
                    if !gpu.scene.loaded {
                        match gpu.scene.rr.try_recv() {
                            Ok(result) => {
                                gpu.update_loaded(result);
                            }
                            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                                panic!("dead thread");
                            }
                            _ => (),
                        }
                    }
                    if self.should_draw {
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
