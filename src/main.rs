use winit::event_loop::EventLoop;
mod app;

fn main() {
    let mut app = app::MyApp::default();
    let event_loop = EventLoop::new().expect("can't create event loop");
    let _ = event_loop.run_app(&mut app);
}
