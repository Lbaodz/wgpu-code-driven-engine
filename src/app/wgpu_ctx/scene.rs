pub mod audio;
pub mod camera;
pub mod meshes;
use crate::app::wgpu_ctx::scene::camera::{LightData, LightDataAlign};
use crate::app::wgpu_ctx::scene::meshes::collision::DoorAndJoint;
use bytemuck::{Pod, Zeroable};
use camera::{Light, LightCtx};
use ctt::encoders::bc7enc::Bc7encSettings;
use ctt::{ConvertSettings, Format, convert, encoders::Encoder};
use ctt::{PipelineOutput, Surface};
use glam::{Mat4, Quat, Vec4, Vec4Swizzles};
use meshes::{
    BakedMeshes, BakedTexture, Door, IsDoor, Meshes, ModelMatrix, Primitive, Texture, Vertex,
};
use pub_fields::pub_fields;
use rapier3d::{
    control::KinematicCharacterController,
    dynamics::{ImpulseJointSet, RigidBodyHandle, RigidBodySet},
    geometry::ColliderSet,
    math::Vec3,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use wgpu::TextureUsages;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{ComputePipeline, RenderPipeline};
pub mod scene_helper;

#[pub_fields]
#[repr(C)]
#[derive(Debug, Default, Pod, Zeroable, Clone, Copy)]
pub struct CacheLights {
    lights_count: u32,
    lights_in_tile: [u32; 15],
}

#[pub_fields]
pub struct AllPipeline {
    render_pipeline: RenderPipeline,
    transparency_pipeline: RenderPipeline,
    early_depth_pipeline: RenderPipeline,
    compute_pipeline: ComputePipeline,
}

#[pub_fields]
pub struct RenderCtx {
    pipeline: AllPipeline,
    camera_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    camera: camera::Camera,
    texture_layout: wgpu::BindGroupLayout,
    mbg_layout: wgpu::BindGroupLayout,
}

#[pub_fields]
pub struct ResultSent {
    meshes: Vec<Meshes>,
    transparency_meshes: Vec<Meshes>,
    impulse_joint: ImpulseJointSet,
    door_joint_handles: HashMap<u32, DoorAndJoint>,
    rbs: RigidBodySet,
    cs: ColliderSet,
    char_handle: RigidBodyHandle,
    char_controller: KinematicCharacterController,
    audio: audio::Audio,
    lights: Vec<Light>,
    light_ctx: LightCtx,
    render_ctx: RenderCtx,
}

#[pub_fields]
pub struct Scene {
    meshes: Vec<Meshes>,
    transparency_meshes: Vec<Meshes>,
    lights: Vec<Light>,
    light_first_loaded: bool,
    rr: Receiver<ResultSent>,
    loaded: bool,
}

impl Scene {
    pub fn create_all_shadow_buffer(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        lights: &Vec<Light>,
    ) -> (wgpu::Buffer, wgpu::Buffer)
     {
        let lights_data: Vec<LightData> = lights.iter()
        .map(|l|
            l.data
        ).collect();
        let cache_buffer = Scene::create_cache_light_buffer(device, config);
        let data_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("light data buffer"),
            contents: bytemuck::cast_slice(&lights_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        (data_buffer, cache_buffer)
    }

    fn create_cache_light_buffer(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::Buffer {
        let x_tiles = (config.width + 15) / 16;
        let y_tiles = (config.height + 15) / 16;
        let total_tiles = (x_tiles * y_tiles) as usize;
        let mut cache_lights: Vec<CacheLights> = Vec::with_capacity(total_tiles);
        cache_lights.push(CacheLights::default());
        println!("{cache_lights:?}");
        device.create_buffer_init(&BufferInitDescriptor {
            label: Some("light cache buffer"),
            contents: bytemuck::cast_slice(&cache_lights),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn create_light_compute_bg(
        device: &wgpu::Device,
        all_lights_buffer: &wgpu::Buffer,
        lights_cache_buffer: &wgpu::Buffer,
    ) -> (wgpu::BindGroup, wgpu::BindGroupLayout) {
        let comp_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light cache comp bg layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        (
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("light cache comp bg"),
                layout: &comp_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: all_lights_buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: lights_cache_buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                ],
            }),
            comp_bg_layout,
        )
    }

    pub fn create_cache_light_bg(
        device: &wgpu::Device,
        all_lights_buffer: &wgpu::Buffer,
        lights_cache_buffer: &wgpu::Buffer
    ) -> (
        wgpu::BindGroup,
        wgpu::BindGroupLayout,
    ) {
        let (c_bg, c_bg_layout) =
            Scene::create_light_compute_bg(device, all_lights_buffer, &lights_cache_buffer);
        (c_bg, c_bg_layout)
    }

    pub fn create_shadow_tt(
        device: &wgpu::Device,
        res_x: u32,
        res_y: u32,
        lights: &Vec<Light>,
    ) -> (Vec<wgpu::TextureView>, wgpu::Sampler, wgpu::TextureView) {
        let tt = device.create_texture(&wgpu::wgt::TextureDescriptor {
            label: Some("light tt arr"),
            size: wgpu::Extent3d {
                width: res_x,
                height: res_y,
                depth_or_array_layers: lights.len() as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = (0..lights.len())
            .map(|i| {
                tt.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("arr tt view {}", i)),
                    format: Some(wgpu::TextureFormat::Depth32Float),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i as u32,
                    array_layer_count: Some(1),
                    aspect: wgpu::TextureAspect::DepthOnly,
                    base_mip_level: 0,
                    mip_level_count: None,
                    ..Default::default()
                })
            })
            .collect();
        let depth_view = tt.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow tt view"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            base_array_layer: 0,
            array_layer_count: Some(lights.len() as u32),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampler shadow"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            compare: Some(wgpu::CompareFunction::GreaterEqual),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        (view, sampler, depth_view)
    }

    pub fn create_light_bindgroup(
        device: &wgpu::Device,
        bindgroup_layout: &wgpu::BindGroupLayout,
        lights: &Vec<Light>,
    ) -> wgpu::BindGroup {
        let all_light_matrices: Vec<LightDataAlign> = lights
            .iter()
            .map(|l| LightDataAlign {
                data: l.data,
                _pad: [0; 36],
            })
            .collect();
        let light_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("light mtx buffer"),
            contents: bytemuck::cast_slice(&all_light_matrices),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light_bindgroup"),
            layout: bindgroup_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &light_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(256),
                }),
            }],
        })
    }

    pub fn create_shadow_bindgroup(
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        s: &wgpu::Sampler,
        all_lights_buffer: &wgpu::Buffer,
        lights_cache_buffer: &wgpu::Buffer,
    ) -> (wgpu::BindGroup, wgpu::BindGroupLayout) {
        let shadow_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_bg_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        (
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow texture/view/lightdata bindgroup"),
                layout: &shadow_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &all_lights_buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(s),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &lights_cache_buffer,
                            offset: 0,
                            size: None,
                        }),
                    }
                ],
            }),
            shadow_layout,
        )
    }

    pub fn draw_light(&mut self, encoder: &mut wgpu::CommandEncoder, light_ctx: &LightCtx) {
        self.light_first_loaded = true;
        for i in 0..self.lights.len() {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("renderpass LIGHT: {}", i)),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &light_ctx.light_views[i],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let planes = self.lights[i].planes;
            render_pass.set_pipeline(&light_ctx.light_pipeline);
            let offset = i as u32 * 256;
            render_pass.set_bind_group(0, &light_ctx.light_bg, &[offset]);
            for mesh in &mut self.meshes {
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                for p in &mut mesh.primitives {
                    if planes.frustum_culling(p.min, p.max) {
                        render_pass.set_bind_group(
                            1,
                            &mesh.bind_group_matrices,
                            &[p.offset_buffer],
                        );
                        render_pass.draw_indexed(p.start..(p.start + p.count), 0, 0..1);
                    }
                }
            }
        }
    }

    pub fn align_light_ids(lights: &mut Vec<Light>) {
        for (id, light) in lights.iter_mut().enumerate() {
            light.data.id[0] = id as f32;
        }
    }
}

fn get_model_matrix(
    node: &gltf::Node,
    parent_matrix: &Mat4,
    door_pos: &mut Vec<Vec3>,
    lock_pos: &mut Vec<Vec3>,
    scales: &mut Vec<Vec3>,
    mut id: &mut Option<u32>,
    is_door: &mut bool,
    door_ids: &mut HashMap<usize, usize>,
    lock_ids: &mut HashMap<usize, usize>,
) -> Mat4 {
    let m: Mat4 = match node.transform() {
        gltf::scene::Transform::Matrix { matrix } => bytemuck::cast(matrix),
        gltf::scene::Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => {
            if let (Some(name), Some(id_num)) = (node.name(), &mut id) {
                if name.to_lowercase().contains("door") {
                    if let Ok(i) = &name[4..].parse::<u32>() {
                        *id_num = *i - 1;
                        door_pos.push(Vec3::new(translation[0], translation[1], translation[2]));
                        scales.push(Vec3::new(scale[0], scale[1], scale[2]));
                        *is_door = true;
                        door_ids.insert(*id_num as usize, door_pos.len() - 1);
                        println!("DOOR: {door_pos:?}");
                    } else {
                        println!("{name} is have invalid id")
                    }
                } else if name.to_lowercase().contains("empty") {
                    if let Ok(i) = &name[5..].parse::<u32>() {
                        *id_num = *i - 1;
                        lock_pos.push(Vec3::new(translation[0], translation[1], translation[2]));
                        lock_ids.insert(*id_num as usize, lock_pos.len() - 1);
                        println!("LOCK: {lock_pos:?}");
                    } else {
                        println!("{name} is have invalid id")
                    }
                } else {
                    *id = None;
                }
            };
            let t: Mat4 =
                Mat4::from_translation(Vec3::new(translation[0], translation[1], translation[2]));
            let r: Mat4 = Mat4::from_quat(Quat::from_xyzw(
                rotation[0],
                rotation[1],
                rotation[2],
                rotation[3],
            ));
            let s: Mat4 = Mat4::from_scale(Vec3::new(scale[0], scale[1], scale[2]));
            let local_mt = t * r * s;
            parent_matrix * local_mt
        }
    };
    m
}

fn convert_aabb_from_matrix(
    min: Vec3,
    max: Vec3,
    matrix: Mat4,
) -> ([f32; 3], [f32; 3], [f32; 3], [f32; 3]) {
    let old_extent = (max - min) * 0.5;
    let old_center = (min + max) * 0.5;
    let center = (matrix * Vec4::new(old_center.x, old_center.y, old_center.z, 1.0)).xyz();
    let (x, y, z) = (
        matrix.x_axis.xyz().abs(),
        matrix.y_axis.xyz().abs(),
        matrix.z_axis.xyz().abs(),
    );

    let extent = x * old_extent.x + y * old_extent.y + z * old_extent.z;

    let min = center - extent;
    let max = center + extent;
    (min.into(), max.into(), center.into(), extent.into())
}

fn read_node(
    node: gltf::Node,
    images: &Vec<gltf::image::Data>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture_layout: &wgpu::BindGroupLayout,
    buffers: &Vec<gltf::buffer::Data>,
    mut all_verticles: &mut Vec<meshes::Vertex>,
    mut all_indices: &mut Vec<u32>,
    mut pris: &mut Vec<meshes::Primitive>,
    mut textures: &mut Vec<meshes::Texture>,
    mut img_cache: &mut HashMap<usize, usize>,
    mut matrices: &mut Vec<meshes::ModelMatrix>,
    parent_matrix: &Mat4,
    mut door_pos: &mut Vec<Vec3>,
    mut lock_pos: &mut Vec<Vec3>,
    mut scales: &mut Vec<Vec3>,
    mut door_ids: &mut HashMap<usize, usize>,
    mut lock_ids: &mut HashMap<usize, usize>,
    mut baked_tt: &mut Vec<meshes::BakedTexture>,
) {
    // door check
    let mut id: Option<u32> = Some(0);
    let mut door = false;

    // matrix model
    let model_matrix = get_model_matrix(
        &node,
        &parent_matrix,
        &mut door_pos,
        &mut lock_pos,
        &mut scales,
        &mut id,
        &mut door,
        &mut door_ids,
        &mut lock_ids,
    );

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            // min max
            let accessor = primitive
                .get(&gltf::Semantic::Positions)
                .expect("no accessor");
            let (mi, ma) = (
                accessor.min().expect("no min"),
                accessor.max().expect("no max"),
            );
            let min_raw: [f32; 3] = serde_json::from_value(mi.clone()).expect("cant extract min");
            let max_raw: [f32; 3] = serde_json::from_value(ma.clone()).expect("cant extract min");
            let (min, max, center, extent) =
                convert_aabb_from_matrix(min_raw.into(), max_raw.into(), model_matrix);

            // texture
            let material = primitive.material();
            let Some(img_info) = material.emissive_texture() else {
                println!("NO EMISSION");
                return;
            };
            let img_index = img_info.texture().source().index();
            let texture_id: usize = *img_cache.entry(img_index).or_insert_with(|| {
                let img_data = &images[img_index];
                let rgba_pixel = match img_data.format {
                    gltf::image::Format::R8G8B8 => {
                        let mut converter: Vec<u8> = Vec::with_capacity(
                            img_data.width as usize * 4 * img_data.height as usize,
                        );
                        for chunk in img_data.pixels.chunks_exact(3) {
                            converter.push(chunk[0]);
                            converter.push(chunk[1]);
                            converter.push(chunk[2]);
                            converter.push(255);
                        }
                        converter
                    }
                    gltf::image::Format::R8G8B8A8 => img_data.pixels.clone(),
                    _ => {
                        panic!("unsupported format texture image")
                    }
                };
                scene_helper::ram("before init bc7");
                let format = Format::BC7_UNORM_BLOCK;
                let encoder = Encoder::Bc7enc(Bc7encSettings::default());
                let convert_settings = ConvertSettings {
                    format: Some(ctt::TargetFormat::Compressed { format, encoder }),
                    container: ctt::Container::Raw,
                    ..Default::default()
                };
                let surface = Surface {
                    data: rgba_pixel,
                    width: img_data.width,
                    height: img_data.height,
                    depth: 1,
                    stride: img_data.width * 4,
                    slice_stride: 0,
                    format: Format::R8G8B8A8_UNORM,
                    color_space: ctt::ColorSpace::Linear,
                    alpha: ctt::AlphaMode::Straight,
                };
                let image = ctt::Image {
                    surfaces: vec![vec![surface]],
                    kind: ctt::TextureKind::Texture2D,
                };
                let bc7_output =
                    convert(image, convert_settings).expect("cant compress rgba to bc7unorm");
                let bc7_data = match bc7_output {
                    PipelineOutput::Raw(mut compressed_image) => {
                        compressed_image
                            .surfaces
                            .pop()
                            .expect("cant pop first")
                            .pop()
                            .expect("cant pop second")
                            .data
                    }
                    _ => {
                        panic!("failed compress")
                    }
                };
                println!("bc7 len in u8 vec: {}", bc7_data.len());
                // wgpu tt process
                let block_x = (img_data.width + 3) / 4 * 4;
                let block_y = (img_data.height + 3) / 4 * 4;
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("tt"),
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Bc7RgbaUnorm,
                    mip_level_count: 1,
                    sample_count: 1,
                    size: wgpu::Extent3d {
                        width: block_x,
                        height: block_y,
                        depth_or_array_layers: 1,
                    },
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                queue.write_texture(
                    texture.as_image_copy(),
                    &bc7_data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(block_x * 4),
                        rows_per_image: Some(block_y),
                    },
                    wgpu::Extent3d {
                        width: block_x,
                        height: block_y,
                        depth_or_array_layers: 1,
                    },
                );
                baked_tt.push(BakedTexture {
                    texture: bc7_data,
                    width: block_x,
                    height: block_y,
                });
                scene_helper::ram("after zip bc7");
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::Repeat,
                    address_mode_v: wgpu::AddressMode::Repeat,
                    address_mode_w: wgpu::AddressMode::Repeat,
                    ..Default::default()
                });
                let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("tt bindgroup"),
                    layout: &texture_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                    ],
                });
                textures.push(Texture {
                    texture: texture_bind_group,
                });

                textures.len() - 1
            });

            // primitive
            let reader = primitive.reader(|b| Some(&buffers[b.index()]));
            let pos: Vec<[f32; 3]> = reader.read_positions().expect("cant get pos").collect();
            let count = pos.len();
            let uv: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .expect("no uv")
                .into_f32()
                .map(|i| i)
                .collect();
            let nor: Vec<[f32; 3]> = reader.read_normals().expect("cant get nor").collect();

            let verticle: Vec<Vertex> = (0..count)
                .map(|i| Vertex {
                    position: pos[i],
                    normal: nor[i],
                    uv: uv[i],
                })
                .collect();

            let offset = all_verticles.len() as u32;
            let start = all_indices.len() as u32;
            let indices: Vec<u32> = reader
                .read_indices()
                .expect("no indices")
                .into_u32()
                .map(|i| i + offset)
                .collect();

            let id = if door { id } else { None };
            let is_door = IsDoor { id, door };

            let offset_buffer = matrices.len() as u32 * 256;
            pris.push(Primitive {
                start: start,
                count: indices.len() as u32,
                min,
                max,
                center,
                extent,
                texture_id,
                offset_buffer,
                is_door,
            });
            // matrices
            matrices.push(ModelMatrix {
                matrix: model_matrix,
                pad: [0; 48],
            });

            all_verticles.extend(verticle);
            all_indices.extend(indices);
        }
    }
    for child in node.children() {
        read_node(
            child,
            &images,
            &device,
            &queue,
            &texture_layout,
            &buffers,
            &mut all_verticles,
            &mut all_indices,
            &mut pris,
            &mut textures,
            &mut img_cache,
            &mut matrices,
            &model_matrix,
            &mut door_pos,
            &mut lock_pos,
            &mut scales,
            door_ids,
            lock_ids,
            &mut baked_tt,
        );
    }
}

fn read_meta_primitive(
    document: &gltf::Document,
    total_vertex: &mut usize,
    total_index: &mut usize,
) {
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            *total_vertex += primitive
                .get(&gltf::Semantic::Positions)
                .map_or(0, |c| c.count());
            *total_index += primitive.indices().map_or(0, |c| c.count())
        }
    }
}

pub fn flat_world_doors<'a>(meshes: &'a mut Vec<Meshes>) {
    let mut offset_id = 0u32;

    'mesh: for mesh in meshes {
        if mesh.primitives.iter().any(|p| p.is_door.door) {
            let doors_len = mesh.doors.len();
            for p in &mut mesh.primitives {
                if let Some(id) = &mut p.is_door.id {
                    *id += offset_id;
                }
            }
            let old_cap = mesh.doors.capacity();
            let old_doors = std::mem::replace(&mut mesh.doors, HashMap::with_capacity(old_cap));
            mesh.doors = old_doors
                .into_iter()
                .map(|(id, door)| (id + offset_id, door))
                .collect();
            println!("{:?}", mesh.doors);

            offset_id += doors_len as u32;
            continue 'mesh;
        }
    }
}

pub fn load_model(
    name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture_layout: &wgpu::BindGroupLayout,
    model_matrix_layout: &wgpu::BindGroupLayout,
) -> Meshes {
    let reload_path = format!("./assets/baked_models/{}.model", name);
    if let Ok(file) = File::open(&reload_path) {
        device.on_uncaptured_error(Arc::new(|error| {
            panic!("WGPU ERROR: {}", error);
        }));

        let mut decoder = zstd::Decoder::new(file).expect("cant create decoder");
        let real_data: BakedMeshes =
            bincode::deserialize_from(decoder.by_ref()).expect("cant deserialize");
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vx_buffer"),
            contents: bytemuck::cast_slice(&real_data.vertex),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("ix_buffer"),
            contents: bytemuck::cast_slice(&real_data.index),
            usage: wgpu::BufferUsages::INDEX,
        });
        // model matrices
        let matrices: Vec<ModelMatrix> = real_data
            .matrices
            .iter()
            .map(|m| ModelMatrix {
                matrix: *m,
                pad: [0; 48],
            })
            .collect();
        let buffer_matrices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matrices buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: matrices.len() as u64 * 256,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer_matrices, 0, bytemuck::cast_slice(&matrices));
        let bind_group_matrices = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_matrices"),
            layout: &model_matrix_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer_matrices,
                    offset: 0,
                    size: std::num::NonZero::new(64),
                }),
            }],
        });
        drop(matrices);

        // tt
        let textures: Vec<Texture> = real_data
            .baked_texture
            .iter()
            .map(|t| {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("tt"),
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Bc7RgbaUnorm,
                    mip_level_count: 1,
                    sample_count: 1,
                    size: wgpu::Extent3d {
                        width: t.width,
                        height: t.height,
                        depth_or_array_layers: 1,
                    },
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                queue.write_texture(
                    texture.as_image_copy(),
                    &t.texture,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(t.width * 4),
                        rows_per_image: Some(t.height),
                    },
                    wgpu::Extent3d {
                        width: t.width,
                        height: t.height,
                        depth_or_array_layers: 1,
                    },
                );
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::Repeat,
                    address_mode_v: wgpu::AddressMode::Repeat,
                    address_mode_w: wgpu::AddressMode::Repeat,
                    ..Default::default()
                });
                let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("tt bindgroup"),
                    layout: &texture_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                    ],
                });
                Texture {
                    texture: texture_bind_group,
                }
            })
            .collect();

        // doors
        let doors = real_data
            .doors
            .iter()
            .map(|(k, d)| {
                (
                    *k,
                    Door {
                        lock_pos: d.lock_pos.into(),
                        lock_for_door: d.lock_for_door.into(),
                        scale: d.scale.into(),
                    },
                )
            })
            .collect();
        Meshes {
            vertex_buffer,
            index_buffer,
            primitives: real_data.primitives,
            textures,
            bind_group_matrices,
            buffer_matrices,
            doors,
        }
    } else {
        let first_load_path = format!("./assets/models/{}.glb", name);
        let (document, buffers, images) = gltf::import(first_load_path).expect("Not found path");
        let nodes_count = document.nodes().len();
        // buffer render offset **IMPORTANT: NO SEPARATE BUFFER FOR EACH RETURN**
        let mut input = (0usize, 0usize);
        read_meta_primitive(&document, &mut input.0, &mut input.1);
        let mut all_verticles: Vec<Vertex> = Vec::with_capacity(input.0);
        let mut all_indices: Vec<u32> = Vec::with_capacity(input.1);
        // index range
        let mut pris: Vec<Primitive> = Vec::with_capacity(nodes_count);
        // tt
        let mut textures: Vec<Texture> = Vec::new();
        let mut img_cache: HashMap<usize, usize> = HashMap::new();
        let mut baked_texture: Vec<BakedTexture> = Vec::new();
        // model matrices
        let mut matrices: Vec<ModelMatrix> = Vec::with_capacity(nodes_count);
        let default_mat = Mat4::IDENTITY;
        // door check
        let mut door_pos: Vec<Vec3> = Vec::new();
        let mut lock_pos: Vec<Vec3> = Vec::new();
        let mut scales: Vec<Vec3> = Vec::new();
        let mut door_ids: HashMap<usize, usize> = HashMap::new();
        let mut lock_ids: HashMap<usize, usize> = HashMap::new();

        for scene in document.scenes() {
            for node in scene.nodes() {
                read_node(
                    node,
                    &images,
                    &device,
                    &queue,
                    &texture_layout,
                    &buffers,
                    &mut all_verticles,
                    &mut all_indices,
                    &mut pris,
                    &mut textures,
                    &mut img_cache,
                    &mut matrices,
                    &default_mat,
                    &mut door_pos,
                    &mut lock_pos,
                    &mut scales,
                    &mut door_ids,
                    &mut lock_ids,
                    &mut baked_texture,
                );
                scene_helper::ram("read node");
            }
        }
        scene_helper::ram("before vt buffer");
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vx_buffer"),
            contents: bytemuck::cast_slice(&all_verticles),
            usage: wgpu::BufferUsages::VERTEX,
        });
        scene_helper::ram("after vt buffer/before index");
        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("ix_buffer"),
            contents: bytemuck::cast_slice(&all_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        scene_helper::ram("after vt buffer/after index/before matrix");
        // model matrices
        let buffer_matrices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matrices buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: matrices.len() as u64 * 256,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer_matrices, 0, bytemuck::cast_slice(&matrices));
        let bind_group_matrices = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_matrices"),
            layout: &model_matrix_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer_matrices,
                    offset: 0,
                    size: std::num::NonZero::new(64),
                }),
            }],
        });
        scene_helper::ram("after vt buffer/after index/after matrix/before door hashmap");

        let mut doors: HashMap<u32, Door> = HashMap::new();
        println!("{:?}, AND_AND_AND {:?}", lock_pos, door_pos);
        if (lock_pos.len() == 0) || (door_pos.len() == 0) {
            for p in &mut pris {
                if p.is_door.door {
                    p.is_door.door = false;
                }
            }
            ()
        } else {
            for id in 0..lock_pos.len() {
                if let Some(door_index) = door_ids.get(&id) {
                    if let Some(lock_index) = lock_ids.get(&id) {
                        doors.insert(
                            id as u32,
                            Door {
                                lock_pos: lock_pos[*lock_index],
                                lock_for_door: lock_pos[*lock_index] - door_pos[*door_index],
                                scale: scales[*door_index],
                            },
                        );
                        println!("{doors:?}");
                    }
                }
            }
        }
        scene_helper::ram(
            "after vt buffer/after index/after matrix/after door hashmap/ before init stream serd and zip zstd",
        );

        let ser_matrices = matrices.iter().map(|m| m.matrix).collect::<Vec<Mat4>>();
        let ser_doors: HashMap<u32, Door> = doors
            .iter()
            .map(|(k, d)| {
                (
                    *k,
                    Door {
                        lock_for_door: d.lock_for_door,
                        lock_pos: d.lock_pos,
                        scale: d.scale,
                    },
                )
            })
            .collect();
        let file = File::create(reload_path.as_str()).expect("cant create file");
        let mut encoder = zstd::Encoder::new(file, 8).expect("cant create encoder");
        scene_helper::ram("encoder");
        bincode::serialize_into(
            encoder.by_ref(),
            &BakedMeshes {
                vertex: all_verticles,
                index: all_indices,
                primitives: pris.clone(),
                baked_texture,
                matrices: ser_matrices,
                doors: ser_doors,
            },
        )
        .expect("cant ser data");
        scene_helper::ram(
            "after vt buffer/after index/after matrix/after door hashmap/ after init stream serd and zip zstd and before finish encoder of zstd",
        );
        encoder.finish().expect("cant finish");
        scene_helper::ram("after it and finish 1 mesh");
        Meshes {
            vertex_buffer,
            index_buffer,
            primitives: pris,
            textures,
            bind_group_matrices,
            buffer_matrices,
            doors,
        }
    }
}
