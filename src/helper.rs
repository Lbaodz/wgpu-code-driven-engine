use egui::{Color32, FontFamily, FontId, Response};
use wgpu::{CommandEncoder, RenderPass, TextureView};

const COLOR_BASE: (u8, u8, u8, u8) = (70, 10, 10, 100);
const COLOR_HOV: (u8, u8, u8, u8) = (150, 15, 15, 210);
const COLOR_CLK: (u8, u8, u8, u8) = (210, 20, 20, 255);

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(
        (a.r() as f32 + (b.r() as f32 - a.r() as f32) * t) as u8,
        (a.g() as f32 + (b.g() as f32 - a.g() as f32) * t) as u8,
        (a.b() as f32 + (b.b() as f32 - a.b() as f32) * t) as u8,
        (a.a() as f32 + (b.a() as f32 - a.a() as f32) * t) as u8,
    )
}

pub fn layout_but_ui(ui: &mut egui::Ui, label: &str, anid: &str) -> Response {
    let (rect, res) = ui.allocate_exact_size(egui::vec2(250.0, 30.0), egui::Sense::click());
    let hover_t = ui.ctx().animate_value_with_time(
        egui::Id::new(anid),
        if res.hovered() { 1.0 } else { 0.0 },
        0.15,
    );
    let click_t = if res.clicked() { 1.0 } else { 0.0 } as f32;

    let bg = lerp_color(
        Color32::from_rgba_unmultiplied(COLOR_BASE.0, COLOR_BASE.1, COLOR_BASE.2, COLOR_BASE.3),
        Color32::from_rgba_unmultiplied(COLOR_HOV.0, COLOR_HOV.1, COLOR_HOV.2, COLOR_HOV.3),
        hover_t,
    );

    let bg = lerp_color(
        bg,
        Color32::from_rgba_unmultiplied(COLOR_CLK.0, COLOR_CLK.1, COLOR_CLK.2, COLOR_CLK.3),
        click_t,
    );

    if ui.rect_contains_pointer(rect) {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }

    if ui.is_rect_visible(rect) {
        ui.painter()
            .rect_filled(rect, egui::CornerRadius::same(4), bg);
        ui.painter().text(
            rect.left_center(),
            egui::Align2::LEFT_CENTER,
            label,
            FontId::new(20.0, FontFamily::Monospace),
            Color32::WHITE,
        );
    };
    res
}

pub fn layout_sld_ui(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    anid: &str,
    range: std::ops::Range<i32>,
) -> (Response, f32) {
    let size_y = 30.0;
    let size_text = size_y - 5.0;
    let size_offset = size_y / 2.0;
    let (slider, res) =
        ui.allocate_exact_size(egui::vec2(400.0, size_y), egui::Sense::click_and_drag());
    let (min, max) = (range.start as f32, range.end as f32);

    if res.dragged() || res.clicked() {
        if let Some(pos) = res.interact_pointer_pos() {
            let cur_ratio = ((pos.x - slider.left()) / slider.width()).clamp(0.0, 1.0);
            *value = (min + cur_ratio * (max - min)).floor();
        }
    }

    let cur_ratio = (*value - min) / (max - min);

    let hover_t = ui.ctx().animate_value_with_time(
        egui::Id::new(anid),
        if res.hovered() || res.dragged() {
            1.0
        } else {
            0.0
        },
        0.15,
    );

    if ui.rect_contains_pointer(slider) {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }

    if ui.is_rect_visible(slider) {
        let painter = ui.painter();

        let tracked = egui::Rect::from_min_size(
            egui::pos2(slider.left(), slider.center().y - size_offset),
            egui::vec2(slider.width(), size_y),
        );

        let bg = lerp_color(
            Color32::from_rgba_unmultiplied(130, 20, 20, 255),
            Color32::from_rgba_unmultiplied(200, 20, 20, 255),
            hover_t,
        );

        painter.text(
            egui::pos2(
                slider.left() + slider.width() * cur_ratio,
                slider.center().y,
            ),
            egui::Align2::LEFT_CENTER,
            label,
            egui::FontId::new(size_text, egui::FontFamily::Monospace),
            bg,
        );

        painter.rect_filled(
            egui::Rect::from_min_size(
                tracked.min,
                egui::vec2(slider.width() * cur_ratio, slider.height()),
            ),
            egui::CornerRadius::same(3),
            bg,
        );
        painter.rect_stroke(
            slider,
            egui::CornerRadius::same(3),
            egui::Stroke::new(1.5, Color32::from_rgba_unmultiplied(230, 20, 20, 200)),
            egui::StrokeKind::Middle,
        );

        painter.text(
            egui::pos2(
                slider.left() + slider.width() * cur_ratio,
                slider.center().y,
            ),
            egui::Align2::RIGHT_CENTER,
            value.clone(),
            egui::FontId::new(size_text, egui::FontFamily::Monospace),
            Color32::WHITE,
        );
    }

    (res, *value)
}

pub fn layout_chb_ui(ui: &mut egui::Ui, label: &str, anid: &str, checked: &bool) -> Response {
    let size_rect = 35.0;
    let offset = size_rect / 5.0;
    let text_size = size_rect - 5.0;
    let (rect, res) =
        ui.allocate_exact_size(egui::vec2(size_rect, size_rect), egui::Sense::click());
    let text_area = egui::Rect::from_center_size(
        egui::pos2(
            rect.left() - text_size * label.len() as f32 / 1.5,
            rect.center().y,
        ),
        rect.size(),
    );

    let hover_t = ui.ctx().animate_value_with_time(
        egui::Id::new(anid),
        if res.hovered() { 1.0 } else { 0.0 },
        0.15,
    );
    let click_t = if *checked { 1.0 } else { 0.0 };

    if ui.rect_contains_pointer(rect) {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }

    let bg = lerp_color(
        Color32::from_rgba_unmultiplied(COLOR_BASE.0, COLOR_BASE.1, COLOR_BASE.2, COLOR_BASE.3),
        Color32::from_rgba_unmultiplied(COLOR_HOV.0, COLOR_HOV.1, COLOR_HOV.2, COLOR_HOV.3),
        hover_t,
    );

    let bg = lerp_color(
        bg,
        Color32::from_rgba_unmultiplied(COLOR_CLK.0, COLOR_CLK.1, COLOR_CLK.2, COLOR_CLK.3),
        click_t,
    );

    let filled_rect = egui::Rect::from_center_size(
        rect.center(),
        egui::vec2(rect.width() - offset, rect.height() - offset),
    );

    if ui.is_rect_visible(rect) {
        let painter = ui.painter();
        painter.text(
            text_area.center(),
            egui::Align2::LEFT_CENTER,
            label,
            egui::FontId::monospace(text_size),
            Color32::WHITE,
        );
        let rad = 0;
        painter.rect_filled(filled_rect, rad, bg);
        painter.rect_stroke(
            rect,
            rad,
            egui::Stroke::new(
                1.0,
                Color32::from_rgba_unmultiplied(COLOR_CLK.0, COLOR_CLK.1, COLOR_CLK.2, COLOR_CLK.3),
            ),
            egui::StrokeKind::Outside,
        );
    }
    ui.add_space(rect.width() + text_area.width());
    res
}

pub fn game_pass<'a>(
    encoder: &'a mut CommandEncoder,
    view: &'a TextureView,
    depth_view: &'a TextureView,
) -> RenderPass<'a> {
    let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("render pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    /* r: 1.0, g: 0.945, b:0.772, a: 1.0 */
                    r: 0.4,
                    g: 0.4,
                    b: 0.4,
                    a: 1.0,
                }),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        ..Default::default()
    });
    render_pass
}

pub fn menu_pass<'a>(
    encoder: &'a mut CommandEncoder,
    view: &'a TextureView,
    depth_view: &'a TextureView,
) -> RenderPass<'a> {
    let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("render pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::DontCare(wgpu::LoadOpDontCare::default()),
                /* ::Clear(wgpu::Color {
                    r: 0.03,
                    g: 0.03,
                    b: 0.03,
                    a: 1.0,
                }), */
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        ..Default::default()
    });
    render_pass
}
