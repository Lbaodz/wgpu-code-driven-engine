use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicBool, AtomicU32}};
use std::sync::mpsc::channel;
use pub_fields::pub_fields;
use winit::window::{CursorGrabMode, Window};
use std::thread;
use game_manager::{GameState, PerformanceState};
use scene::{ResultSent, meshes::Meshes};
use rapier3d::{
    dynamics::{ImpulseJointSet,RigidBodySet},
    geometry::ColliderSet
};
use scene::audio;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;
use std::sync::Mutex;
pub mod scene;
pub mod ui;
pub mod game_manager;
pub mod wgpu_helper;
use ui::ui_helper;
use scene::meshes;

#[pub_fields] 
pub struct WgpuCtx {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    camera_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    camera: scene::camera::Camera,
    camera_planes: scene::camera::Planes,
    texture_layout: wgpu::BindGroupLayout,
    mbg_layout: wgpu::BindGroupLayout,
    scene: scene::Scene,
    depth_view: wgpu::TextureView,
    collision: scene::meshes::collision::Collision,
    ui: ui::UI,
    game_state: game_manager::GameState,
    playing: bool,
    mouse_locked: bool,
    fps_state: game_manager::PerformanceState,
    audio: audio::Audio,
    file_manager: HashMap<String, game_manager::FileManager>,
    should_load: Arc<AtomicBool>,
    game_level: game_manager::GameLevel,
    loader_progress: Arc<AtomicU32>,
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

    pub fn update_ui_menu(
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
                        .corner_radius(0),
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
                    if ui_helper::layout_but_ui(egui, "Play", "play_but").clicked() {
                        self.game_state = GameState::Play;
                        self.playing = true;
                        win.set_cursor_visible(false);
                        let _ = win.set_cursor_grab(CursorGrabMode::Locked);
                        self.mouse_locked = true;
                    };
                    // set
                    if ui_helper::layout_but_ui(egui, "Settings", "settings_but").clicked() {
                        self.game_state = GameState::Settings
                    };
                    egui.add_space(80.0);
                    // ext
                    if ui_helper::layout_but_ui(egui, "Exit", "exit_but").clicked() {
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

    pub fn update_ui_settings(
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
                        .corner_radius(0),
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
                        ui_helper::layout_sld_ui(egui, "Fov", &mut self.camera.fov, "fov", 30..120);
                    if fov.clicked() || fov.drag_stopped() {
                        self.camera.fov = fov_val;
                        self.camera.is_moving = true;
                    };
                    egui.add_space(5.0);
                    // sound (deadcode)
                    let (sound, sound_val) = ui_helper::layout_sld_ui(
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
                        if ui_helper::layout_chb_ui(ui, "30", "30_chb", &self.fps_state.low).clicked()
                        {
                            self.fps_state = PerformanceState {
                                low: !self.fps_state.low,
                                ..Default::default()
                            };
                            fps = 30.0;
                        };
                        if ui_helper::layout_chb_ui(ui, "60", "60_chb", &self.fps_state.mid).clicked()
                        {
                            self.fps_state = PerformanceState {
                                mid: !self.fps_state.mid,
                                ..Default::default()
                            };
                            fps = 60.0;
                        };
                        if ui_helper::layout_chb_ui(ui, "90", "90_chb", &self.fps_state.ok).clicked() {
                            self.fps_state = PerformanceState {
                                ok: !self.fps_state.ok,
                                ..Default::default()
                            };
                            fps = 90.0;
                        };
                        if ui_helper::layout_chb_ui(ui, "120", "120_chb", &self.fps_state.high)
                            .clicked()
                        {
                            self.fps_state = PerformanceState {
                                high: !self.fps_state.high,
                                ..Default::default()
                            };
                            fps = 120.0;
                        };
                        if ui_helper::layout_chb_ui(ui, "144", "144_chb", &self.fps_state.very_high)
                            .clicked()
                        {
                            self.fps_state = PerformanceState {
                                very_high: !self.fps_state.very_high,
                                ..Default::default()
                            };
                            fps = 144.0;
                        };
                        if ui_helper::layout_chb_ui(ui, "240", "240_chb", &self.fps_state.epic)
                            .clicked()
                        {
                            self.fps_state = PerformanceState {
                                epic: !self.fps_state.epic,
                                ..Default::default()
                            };
                            fps = 240.0;
                        };
                        if self.fps_state.all_false() {
                            if ui_helper::layout_chb_ui(ui, "???", "???", &true).clicked() {
                                ()
                            };
                        };
                    });
                    egui.add_space(40.0);
                    // ext
                    if ui_helper::layout_but_ui(egui, "Back", "back_but").clicked() {
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

    pub fn update_ui_loading(
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
                        .corner_radius(0),
                )
                .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(-25.0, -25.0))
                .show(&ui.egui_ctx, |egui| {
                    let progress = self
                        .loader_progress
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let label = format!("Loading {}%", progress);
                    egui.style_mut().interaction.selectable_labels = false;
                    egui.add_space(50.0);
                    egui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(label)
                                .font(egui::FontId::monospace(50.0))
                                .strong()
                                .color(egui::Color32::from_rgba_unmultiplied(210, 20, 20, 180)),
                        );
                        ui_helper::layout_prg_ui(ui, "progress", &progress);
                    });
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

    pub fn load_all(&mut self, scene_name: String) {
        self.should_load
            .store(false, std::sync::atomic::Ordering::Relaxed);
        let (sd, rr) = channel::<ResultSent>();
        self.scene.rr = rr;
        let (
            m_paths,
            a_paths,
            texture_layout_clone,
            mbg_layout_clone,
            mut impulse_joint,
            mut joints_handle,
            mut doors_handle,
            mut rbs,
            mut cs,
            progress_clone,
            device,
            queue,
            pos_cam,
        ) = (
            self.file_manager
                .get(&scene_name)
                .expect("no scene name")
                .model_paths
                .clone(),
            self.file_manager
                .get(&scene_name)
                .expect("no scene name")
                .audio_paths
                .clone(),
            self.texture_layout.clone(),
            self.mbg_layout.clone(),
            ImpulseJointSet::new(),
            Vec::new(),
            Vec::new(),
            RigidBodySet::new(),
            ColliderSet::new(),
            Arc::clone(&self.loader_progress),
            self.device.clone(),
            self.queue.clone(),
            self.camera.eye.into(),
        );

        thread::spawn(move || {
            let counter = Instant::now();
            let mut progress_counter: f32 = 0.0;
            let mut path_buf = String::with_capacity(64);

            let mut audio = audio::Audio::new();
            let progress_audio: f32 = 20.0 / a_paths.len() as f32;
            for path in a_paths {
                path_buf.clear();
                path_buf.push_str("./assets/audio/");
                path_buf.push_str(&path);
                let Some(path_name) = Path::new(&path).file_stem() else {
                    panic!("no path")
                };
                let Some(name) = path_name.to_str() else {
                    panic!("no path name")
                };
                audio.load(name, &path_buf);
                progress_counter += progress_audio;
                progress_clone.store(
                    progress_counter.floor() as u32,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            println!("Loaded radio in {}ms", counter.elapsed().as_millis());

            let mutex_meshes: Mutex<Vec<Meshes>> = Mutex::new(Vec::with_capacity(m_paths.len()));
            let progress_mesh = 70.0 / m_paths.len() as f32;
            m_paths.par_iter().for_each(|path| {
                let mesh = scene::load_model(
                    &path,
                    &device,
                    &queue,
                    &texture_layout_clone,
                    &mbg_layout_clone,
                );
                let mut lock = mutex_meshes.lock().expect("cant get lock");
                lock.push(mesh);
                progress_clone.store(
                    (progress_counter + progress_mesh).floor() as u32,
                    std::sync::atomic::Ordering::Relaxed,
                );
            });

            let meshes = mutex_meshes.into_inner().expect("no mesh found");
            let progress_physics = 10.0 / meshes.len() as f32;
            for mesh in &meshes {
                for primitive in &mesh.primitives {
                    if !primitive.is_door.door {
                        meshes::load_static_collider(&primitive, &mut rbs, &mut cs);
                    }
                    if primitive.is_door.door && !(mesh.doors.len() == 0) {
                        if let Some(i) = primitive.is_door.id {
                            let id = i as usize;
                            let (door_handle, joint_handle) =
                            meshes::load_door_collider(
                                primitive,
                                &mut rbs,
                                &mut cs,
                                &mesh.doors,
                                id,
                                &mut impulse_joint,
                            );
                            doors_handle.push(door_handle);
                            joints_handle.push(joint_handle);
                        }
                    }
                }
                progress_counter += progress_physics;
                progress_clone.store(
                    progress_counter.floor() as u32,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }

            let (char_handle, char_controller) = meshes::load_player_collision(&pos_cam, &mut rbs, &mut cs);

            let _ = sd.send(ResultSent {
                meshes,
                impulse_joint: impulse_joint,
                doors_handle: doors_handle,
                joints_handle: joints_handle,
                rbs: rbs,
                cs: cs,
                char_handle,
                char_controller,
                audio,
            });
            println!("Loaded in {}ms", counter.elapsed().as_millis());
        });
    }
}