use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};
struct WgpuCtx {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
}

#[derive(Default)]
struct MyApp {
    window: Option<Arc<Window>>,
    gpu_ctx: Option<WgpuCtx>,
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
            WgpuCtx {
                surface,
                device,
                queue,
                config,
            }
        });

        self.window = Some(window);
        self.gpu_ctx = Some(gpu_ctx);
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
                        let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
                    }
                    gpu.queue.submit(std::iter::once(encoder.finish()));
                    frame.present();
                }
                    win.request_redraw();
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
}

fn main() {
    let mut app = MyApp::default();
    let event_loop = EventLoop::new().unwrap();
    let _ = event_loop.run_app(&mut app);
}
