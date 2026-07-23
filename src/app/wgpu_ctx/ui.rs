pub mod ui_helper;
use pub_fields::pub_fields;
#[pub_fields] 
pub struct UI {
    egui_ctx: egui::Context,
    egui_winit_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}