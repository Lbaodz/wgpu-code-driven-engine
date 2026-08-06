use game_manager::{GameState, PerformanceState};
use glam::Vec3;
use pub_fields::pub_fields;
use rapier3d::{
    dynamics::{ImpulseJointSet, RigidBodySet},
    geometry::ColliderSet,
};
use rayon::prelude::*;
use scene::audio;
use scene::{ResultSent, meshes::Meshes};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::sync::mpsc::channel;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU32},
};
use std::thread;
use std::time::Instant;
use winit::window::{CursorGrabMode, Window};
pub mod game_manager;
pub mod scene;
pub mod ui;
pub mod wgpu_helper;
use crate::{
    app::wgpu_ctx::{
        scene::{AllPipeline, camera::Light, meshes::collision::DoorAndJoint},
        wgpu_helper::{
            create_compute_pipeline, create_early_depth_pipeline, create_light_pipeline,
        },
    },
    create_pp_layout, v3,
};
use scene::{
    RenderCtx, Scene,
    camera::{Camera, Planes, UniformCamera},
};
use scene::{camera::LightCtx, meshes};
use ui::ui_helper;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

#[pub_fields]
pub struct WgpuCtx {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_ctx: Option<RenderCtx>,
    light_ctx: Option<LightCtx>,
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

    pub fn bind_basic_bg(
        render_pass: &mut wgpu::RenderPass,
        render_ctx: &RenderCtx,
        light_ctx: &LightCtx,
    ) {
        render_pass.set_bind_group(0, &render_ctx.camera_bind_group, &[]);
        render_pass.set_bind_group(3, &light_ctx.shadow_bg, &[]);
    }

    pub fn draw_meshes(
        &self,
        meshes: &Vec<Meshes>,
        render_pass: &mut wgpu::RenderPass,
        with_tt: bool,
    ) {
        let Some(render_ctx) = &self.render_ctx else {
            panic!("cant get renderctx")
        };

        for mesh in meshes {
            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            for primitive in &mesh.primitives {
                if render_ctx
                    .camera
                    .take_plane()
                    .frustum_culling(primitive.min, primitive.max)
                {
                    if with_tt {
                        render_pass.set_bind_group(
                            1,
                            &mesh.textures[primitive.texture_id].texture,
                            &[],
                        );
                        render_pass.set_bind_group(
                            2,
                            &mesh.bind_group_matrices,
                            &[primitive.offset_buffer],
                        );
                    } else {
                        render_pass.set_bind_group(
                            1,
                            &mesh.bind_group_matrices,
                            &[primitive.offset_buffer],
                        );
                    }
                    render_pass.draw_indexed(
                        primitive.start..(primitive.start + primitive.count),
                        0,
                        0..1,
                    );
                }
            }
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
                    let Some(rd_ctx) = &mut self.render_ctx else {
                        panic!("cant get render ctx");
                    };
                    let (fov, fov_val) = ui_helper::layout_sld_ui(
                        egui,
                        "Fov",
                        &mut rd_ctx.camera.fov,
                        "fov",
                        30..120,
                    );
                    if fov.clicked() || fov.drag_stopped() {
                        rd_ctx.camera.fov = fov_val;
                        rd_ctx.camera.is_moving = true;
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
                        if ui_helper::layout_chb_ui(ui, "30", "30_chb", &self.fps_state.low)
                            .clicked()
                        {
                            self.fps_state = PerformanceState {
                                low: !self.fps_state.low,
                                ..Default::default()
                            };
                            fps = 30.0;
                        };
                        if ui_helper::layout_chb_ui(ui, "60", "60_chb", &self.fps_state.mid)
                            .clicked()
                        {
                            self.fps_state = PerformanceState {
                                mid: !self.fps_state.mid,
                                ..Default::default()
                            };
                            fps = 60.0;
                        };
                        if ui_helper::layout_chb_ui(ui, "90", "90_chb", &self.fps_state.ok)
                            .clicked()
                        {
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
        self.scene.loaded = false;
        let (sd, rr) = channel::<ResultSent>();
        self.scene.rr = rr;
        let file_manager = &self
            .file_manager
            .get(&scene_name)
            .expect("cant get file manager");
        let (
            m_paths,
            t_paths,
            a_paths,
            mut impulse_joint,
            mut door_joint_handles,
            mut rbs,
            mut cs,
            progress_clone,
            device,
            queue,
            config,
        ) = (
            file_manager.model_paths.clone(),
            file_manager.transparency_paths.clone(),
            file_manager.audio_paths.clone(),
            ImpulseJointSet::new(),
            HashMap::new(),
            RigidBodySet::new(),
            ColliderSet::new(),
            Arc::clone(&self.loader_progress),
            self.device.clone(),
            self.queue.clone(),
            self.config.clone(),
        );

        thread::Builder::new()
            .name(format!("thread loader for scene {}", scene_name))
            .spawn(move || {
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

                let texture_layout =
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("tt layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float {
                                        filterable: true,
                                    },
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

                let mut camera = Camera {
                    eye: (0.0, 2.7, 5.0).into(),
                    target: (0.0, 0.0, 0.0).into(),
                    up: Vec3::Y,
                    aspect: config.width as f32 / config.height as f32,
                    fov: 75.0,
                    near: 0.01,
                    yaw: -90.0,
                    pitch: 0.0,
                    is_moving: true,
                    is_rotating: true,
                    planes: None,
                };

                let pos_cam: [f32; 3] = camera.eye.into();

                let mutex_meshes: Mutex<Vec<Meshes>> =
                    Mutex::new(Vec::with_capacity(m_paths.len()));
                let progress_mesh = 40.0 / m_paths.len() as f32;
                let progress_mesh_step = (progress_mesh as u32).max(1);
                m_paths.par_iter().for_each(|path| {
                    let mesh = scene::load_model(
                        &path,
                        &device,
                        &queue,
                        &texture_layout,
                        &model_bind_group_layout,
                    );
                    let mut lock = mutex_meshes.lock().expect("cant get lock");
                    lock.push(mesh);
                    progress_clone
                        .fetch_add(progress_mesh_step, std::sync::atomic::Ordering::Relaxed);
                });

                let mutex_transparency: Mutex<Vec<Meshes>> =
                    Mutex::new(Vec::with_capacity(t_paths.len()));
                let progress_t = 30.0 / t_paths.len() as f32;
                let progress_t_step = (progress_t as u32).max(1);
                t_paths.par_iter().for_each(|path| {
                    let mesh = scene::load_model(
                        &path,
                        &device,
                        &queue,
                        &texture_layout,
                        &model_bind_group_layout,
                    );
                    let mut lock = mutex_transparency.lock().expect("cant get lock");
                    lock.push(mesh);
                    progress_clone.fetch_add(progress_t_step, std::sync::atomic::Ordering::Relaxed);
                });
                let transparency_meshes = mutex_transparency
                    .into_inner()
                    .expect("no transparency found");

                let mut meshes = mutex_meshes.into_inner().expect("no mesh found");
                scene::flat_world_doors(&mut meshes);
                let progress_physics = 10.0 / meshes.len() as f32;
                let progress_physics_step = (progress_physics as u32).max(1);
                for mesh in &meshes {
                    for primitive in &mesh.primitives {
                        if !primitive.is_door.door {
                            meshes::load_static_collider(&primitive, &mut rbs, &mut cs);
                        } else if primitive.is_door.door && !(mesh.doors.len() == 0) {
                            if let Some(i) = primitive.is_door.id {
                                let id = i as usize;
                                let (door_handle, joint_handle, id_door) =
                                    meshes::load_door_collider(
                                        &primitive,
                                        &mut rbs,
                                        &mut cs,
                                        &mesh.doors,
                                        id,
                                        &mut impulse_joint,
                                    );
                                door_joint_handles
                                    .entry(id_door)
                                    .or_insert_with(|| DoorAndJoint {
                                        door_handle,
                                        joint_handle,
                                    });
                            }
                        }
                    }
                    progress_clone
                        .fetch_add(progress_physics_step, std::sync::atomic::Ordering::Relaxed);
                }

                let (char_handle, char_controller) =
                    meshes::load_player_collision(&pos_cam, &mut rbs, &mut cs);

                let light_layout =
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("light_bg_layout"),
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: true,
                                min_binding_size: std::num::NonZeroU64::new(256),
                            },
                            count: None,
                        }],
                    });

                let light_shader =
                    device.create_shader_module(wgpu::include_wgsl!("../shaders/light.wgsl"));
                let pipeline_light_layout =
                    create_pp_layout!(&device, [&light_layout, &model_bind_group_layout]);
                let light_pipeline =
                    create_light_pipeline(&device, &pipeline_light_layout, &light_shader);

                let mut lights = vec![
                    Light::new_spot_light(
                        v3!(-25.0, 2.0, -4.0),
                        v3!(0.5, 0.1, 0.2),
                        45.0,
                        0.01,
                        2.0,
                        [0.6, 0.6, 0.5, 1.0],
                        10.0,
                    ),
                    Light::new_spot_light(
                        v3!(25.0, 2.0, 4.0),
                        v3!(-0.5, 0.1, -0.2),
                        45.0,
                        0.01,
                        2.0,
                        [0.6, 0.6, 0.5, 1.0],
                        10.0,
                    ),
                ];
                Scene::align_light_ids(&mut lights);
                let (data_lights_buffer, cache_lights_buffer) = Scene::create_all_shadow_buffer(&device, &config, &lights);
                let light_tt: (Vec<wgpu::TextureView>, wgpu::Sampler, wgpu::TextureView) =
                    scene::Scene::create_shadow_tt(&device, 1024, 1024, &lights);

                let (shadow_bg, shadow_layout) =
                    Scene::create_shadow_bindgroup(&device, &light_tt.2, &light_tt.1, &data_lights_buffer, &cache_lights_buffer);

                let (
                    compute_lights_bg,
                    compute_lights_bg_layout,
                ) = Scene::create_cache_light_bg(&device, &data_lights_buffer, &cache_lights_buffer);

                let light_ctx = LightCtx::new(
                    Scene::create_light_bindgroup(&device, &light_layout, &lights),
                    light_tt.0,
                    shadow_bg,
                    light_tt.2,
                    light_pipeline,
                    compute_lights_bg,
                );

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

                let camera_uniform = camera.make_camera();
                let planes = Planes::build_plane_from_matrix4(camera_uniform);
                camera.planes = Some(planes);
                let camera_uniform = UniformCamera {
                    uniform: camera_uniform,
                };

                let camera_buffer = device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("cmr_buffer"),
                    contents: bytemuck::bytes_of(&camera_uniform.uniform),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

                let shader =
                    device.create_shader_module(wgpu::include_wgsl!("../shaders/main.wgsl"));
                let early_depth_shader =
                    device.create_shader_module(wgpu::include_wgsl!("../shaders/early_depth.wgsl"));

                let pipeline_basic_layout = create_pp_layout!(
                    &device,
                    [
                        &camera_bind_group_layout,
                        &texture_layout,
                        &model_bind_group_layout,
                        &shadow_layout,
                    ]
                );

                let early_depth_layout = create_pp_layout!(
                    &device,
                    [&camera_bind_group_layout, &model_bind_group_layout]
                );
                let early_depth_pipeline =
                    create_early_depth_pipeline(&device, &early_depth_layout, &early_depth_shader);

                let render_pipeline = wgpu_helper::create_basic_pipeline(
                    &device,
                    &pipeline_basic_layout,
                    &shader,
                    &config,
                );

                let transparency_pipeline = wgpu_helper::create_transparency_pipeline(
                    &device,
                    &pipeline_basic_layout,
                    &shader,
                    &config,
                );

                let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("camera bind group"),
                    layout: &camera_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    }],
                });

                let compute_shader =
                    device.create_shader_module(wgpu::include_wgsl!("../shaders/c_shader.wgsl"));
                let compute_layout = create_pp_layout!(
                    &device,
                    [&camera_bind_group_layout, &compute_lights_bg_layout,]
                );
                let compute_pipeline =
                    create_compute_pipeline(&device, &compute_layout, &compute_shader);

                let pipeline = AllPipeline {
                    render_pipeline,
                    transparency_pipeline,
                    early_depth_pipeline,
                    compute_pipeline,
                };

                let render_ctx = RenderCtx {
                    pipeline,
                    camera_bind_group,
                    camera_buffer,
                    camera,
                    mbg_layout: model_bind_group_layout,
                    texture_layout,
                };

                let _ = sd.send(ResultSent {
                    meshes,
                    transparency_meshes,
                    impulse_joint: impulse_joint,
                    door_joint_handles,
                    rbs: rbs,
                    cs: cs,
                    char_handle,
                    char_controller,
                    audio,
                    lights,
                    light_ctx,
                    render_ctx,
                });
                println!("Loaded all in {}ms", counter.elapsed().as_millis());
            })
            .expect("cant create thread");
    }

    pub fn update_loaded(&mut self, result: ResultSent) {
        let collision = &mut self.collision;
        self.game_state = GameState::Menu;
        self.scene.meshes = result.meshes;
        self.scene.transparency_meshes = result.transparency_meshes;
        self.scene.lights = result.lights;
        collision.rbs = result.rbs;
        collision.cs = result.cs;
        collision.impulse_joint = result.impulse_joint;
        collision.door_joint_handles = result.door_joint_handles;
        collision.char_controller = result.char_controller;
        collision.char_handle = result.char_handle;
        self.audio = result.audio;
        self.render_ctx = Some(result.render_ctx);
        self.light_ctx = Some(result.light_ctx);
        self.scene.loaded = true;
    }
}
