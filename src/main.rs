use std::sync::Arc;
use cgmath::{prelude, Point3, Matrix4, Vector3, Deg};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};
use wgpu::util::{DeviceExt, BufferInitDescriptor};
use std::time::{Duration, Instant};

struct Camera {
    eye: Point3<f32>,
    target: Point3<f32>,
    up: Vector3<f32>,
    aspect: f32,
    fov: f32,
    near: f32,
    far: f32,
}

impl Camera {
    fn make_camera(&self) -> Matrix4<f32> {
        let view = Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(Deg(self.fov), self.aspect, self.near, self.far);
        let wgpu_matrix_correction = Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.5,
            0.0, 0.0, 0.0, 1.0,
        );
        wgpu_matrix_correction * proj * view
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UniformCamera {
    uniform: [[f32;4]; 4]
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

const VERTICES: &[Vertex] = &[
    Vertex { position: [-5.0, 0.0, -5.0], color: [1.0, 0.0, 0.0], },
    Vertex { position: [5.0, 0.0, -5.0], color: [0.0, 1.0, 0.0], },
    Vertex { position: [5.0, 0.0, 5.0], color: [0.0, 0.0, 1.0], },
    Vertex { position: [5.0, 0.0, 5.0], color: [0.0, 0.0, 1.0], },
    Vertex { position: [-5.0, 0.0, 5.0], color: [1.0, 1.0, 0.0], },
    Vertex { position: [-5.0, 0.0, -5.0], color: [1.0, 0.0, 0.0], },
];
struct WgpuCtx {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
}

#[derive(Default)]
struct MyApp {
    window: Option<Arc<Window>>,
    gpu_ctx: Option<WgpuCtx>,
    last_time: Option<Instant>,
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
            },
            _ => None,
        }
    }
}

impl ApplicationHandler for MyApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let win_attrs = Window::default_attributes().with_title("My App");
        let window = Arc::new(event_loop.create_window(win_attrs).unwrap());

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

            let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor { 
                label: Some("vx_buffer"), 
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let camera = Camera {
                eye: (0.0, 1.0, 5.0).into(),
                target: (0.0, 0.0, 0.0).into(),
                up: Vector3::unit_y(),
                aspect: config.width as f32 / config.height as f32,
                fov: 75.0,
                near: 0.1,
                far: 100.0
            };

            let camera_uniform = UniformCamera {
                uniform: camera.make_camera().into(),
            };

            let camera_buffer = device.create_buffer_init(&BufferInitDescriptor {
                label: Some("cmr_buffer"),
                contents: bytemuck::cast_slice(&camera_uniform.uniform),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let shader = device.create_shader_module(wgpu::include_wgsl!("test.wgsl"));

            let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { 
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
                        array_stride: 24,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![
                            0 => Float32x3,
                            1 => Float32x3,
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
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

            let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { 
                label: Some("camera bind group"),
                layout: &camera_bind_group_layout,
                entries: &[wgpu::BindGroupEntry{
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }],
            });

            WgpuCtx {
                surface,
                device,
                queue,
                config,
                pipeline,
                vertex_buffer,
                camera_bind_group,
            }
        });

        self.window = Some(window);
        self.gpu_ctx = Some(gpu_ctx);
        self.last_time = Some(Instant::now());
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::RedrawRequested => {
                if let (Some(win), Some(gpu)) = (&self.window, &self.gpu_ctx) {
                    if let Some((frame, view)) = gpu.get_frame_view() {

                        let mut encoder = gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Encoder"),
                        });

                        {
                            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
                                ..Default::default()
                            });
                            render_pass.set_pipeline(&gpu.pipeline);
                            render_pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
                            render_pass.set_bind_group(0,&gpu.camera_bind_group, &[]);
                            render_pass.draw(0..VERTICES.len() as u32, 0..1);
                        }
                        gpu.queue.submit(std::iter::once(encoder.finish()));
                        frame.present();
                    }
                } 
                else {
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
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let (Some(win), Some(last_time)) = (&self.window, &self.last_time) {
            let desire_time = Duration::from_secs_f64(1.0 / 60.0);
            let time_eslaped = last_time.elapsed();
            if time_eslaped < desire_time {
                let sleep_time = desire_time - time_eslaped;
                std::thread::sleep(sleep_time);
            };
            self.last_time = Some(Instant::now());
            win.request_redraw();
        };
    }
}

fn main() {
    let mut app = MyApp::default();
    let event_loop = EventLoop::new().unwrap();
    let _ = event_loop.run_app(&mut app);
}
