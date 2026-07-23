use wgpu::{CommandEncoder, RenderPass, TextureView};

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

pub fn make_depth_tt(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        label: Some("depth tt"),
        mip_level_count: 1,
        sample_count: 1,
        size: wgpu::Extent3d {
            height: config.height,
            width: config.width,
            depth_or_array_layers: 1,
        },
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[wgpu::TextureFormat::Depth32Float],
    })
}