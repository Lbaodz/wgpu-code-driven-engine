pub mod meshes;
pub mod camera;
pub mod audio;
use std::collections::HashMap;
use std::sync::mpsc::Receiver;
use cgmath::{Quaternion, Matrix4,SquareMatrix, Vector3, Vector4};
use ctt::encoders::bc7enc::Bc7encSettings;
use ctt::{ConvertSettings, Format, convert, encoders::Encoder};
use ctt::{PipelineOutput, Surface};
use rapier3d::{
    control::KinematicCharacterController,
    dynamics::{
        ImpulseJointSet, RigidBodySet, ImpulseJointHandle, RigidBodyHandle
    },
    geometry::{
        ColliderSet
    },
    math::Vec3,
};
use meshes::{Vertex, BakedMeshes, Texture, Meshes, ModelMatrix, IsDoor, Primitive, Door, BakedTexture, BakedDoor};
use std::fs;
use wgpu::util::{DeviceExt, BufferInitDescriptor};
use pub_fields::pub_fields;
pub mod scene_helper;

#[pub_fields] 
pub struct ResultSent {
    meshes: Vec<meshes::Meshes>,
    impulse_joint: ImpulseJointSet,
    doors_handle: Vec<RigidBodyHandle>,
    joints_handle: Vec<ImpulseJointHandle>,
    rbs: RigidBodySet,
    cs: ColliderSet,
    char_handle: RigidBodyHandle,
    char_controller: KinematicCharacterController,
    audio: audio::Audio,
}

#[pub_fields] 
pub struct Scene {
    meshes: Vec<meshes::Meshes>,
    rr: Receiver<ResultSent>,
}

fn get_model_matrix(
    node: &gltf::Node,
    parent_matrix: &Matrix4<f32>,
    door_pos: &mut Vec<Vec3>,
    lock_pos: &mut Vec<Vec3>,
    scales: &mut Vec<Vector3<f32>>,
    mut id: &mut Option<u32>,
    is_door: &mut bool,
) -> [[f32; 4]; 4] {
    let m = match node.transform() {
        gltf::scene::Transform::Matrix { matrix } => matrix,
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
                        scales.push(Vector3::new(scale[0], scale[1], scale[2]));
                        *is_door = true;
                    } else {
                        println!("{name} is have invalid id")
                    }
                } else if name.to_lowercase().contains("empty") {
                    if let Ok(i) = &name[5..].parse::<u32>() {
                        *id_num = *i - 1;
                        lock_pos.push(Vec3::new(translation[0], translation[1], translation[2]));
                    } else {
                        println!("{name} is have invalid id")
                    }
                } else {
                    *id = None;
                }
            };
            let t: Matrix4<f32> = Matrix4::from_translation(Vector3::new(
                translation[0],
                translation[1],
                translation[2],
            ));
            let r: Matrix4<f32> = Matrix4::from(Quaternion::new(
                rotation[3],
                rotation[0],
                rotation[1],
                rotation[2],
            ));
            let s: Matrix4<f32> = Matrix4::from_nonuniform_scale(scale[0], scale[1], scale[2]);
            let local_mt = t * r * s;
            (parent_matrix * local_mt).into()
        }
    };
    m
}

fn convert_aabb_from_matrix(
    min: Vector3<f32>,
    max: Vector3<f32>,
    matrix: Matrix4<f32>,
) -> ([f32; 3], [f32; 3], [f32; 3], [f32; 3]) {
    let old_extent = (max - min) * 0.5;
    let old_center = (min + max) * 0.5;
    let center4 = matrix * Vector4::new(old_center.x, old_center.y, old_center.z, 1.0);
    let center = Vector3::new(center4.x, center4.y, center4.z);

    let mut extent = Vector3::new(0.0, 0.0, 0.0);
    for i in 0..3 {
        for j in 0..3 {
            extent[i] += matrix[j][i] * old_extent[i];
        }
    }
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
    parent_matrix: &Matrix4<f32>,
    mut door_pos: &mut Vec<Vec3>,
    mut lock_pos: &mut Vec<Vec3>,
    mut scales: &mut Vec<Vector3<f32>>,
    mut ids: &mut HashMap<usize, usize>,
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
    );
    if let Some(id) = id {
        let len_ids = ids.len();
        ids.entry(id as usize).or_insert_with(|| len_ids);
    };

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            // min max
            let accessor = primitive.get(&gltf::Semantic::Positions).unwrap();
            let (mi, ma) = (accessor.min().unwrap(), accessor.max().unwrap());
            let min_raw: [f32; 3] = serde_json::from_value(mi.clone()).expect("cant extract min");
            let max_raw: [f32; 3] = serde_json::from_value(ma.clone()).expect("cant extract min");
            let (min, max, center, extent) =
                convert_aabb_from_matrix(min_raw.into(), max_raw.into(), model_matrix.into());

            // texture
            let material = primitive.material();
            let img_info = material.emissive_texture().expect("no info");
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
                        compressed_image.surfaces.pop().unwrap().pop().unwrap().data
                    }
                    _ => {
                        panic!("failed compress")
                    }
                };
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
                .unwrap()
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
            &model_matrix.into(),
            &mut door_pos,
            &mut lock_pos,
            &mut scales,
            &mut ids,
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

pub fn load_model(
    name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture_layout: &wgpu::BindGroupLayout,
    model_matrix_layout: &wgpu::BindGroupLayout,
) -> Meshes {
    let reload_path = format!("./assets/baked_models/{}.model", name);
    if let Ok(file_byte) = fs::read(&reload_path) {
        let real_data = bincode::deserialize::<BakedMeshes>(&file_byte).expect("cant deserialize");
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
        let default_mat = Matrix4::identity();
        // door check
        let mut door_pos: Vec<Vec3> = Vec::new();
        let mut lock_pos: Vec<Vec3> = Vec::new();
        let mut scales: Vec<Vector3<f32>> = Vec::new();
        let mut ids: HashMap<usize, usize> = HashMap::new();

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
                    &mut ids,
                    &mut baked_texture,
                );
                scene_helper::ram();
            }
        }
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vx_buffer"),
            contents: bytemuck::cast_slice(&all_verticles),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("ix_buffer"),
            contents: bytemuck::cast_slice(&all_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
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

        let mut doors: HashMap<u32, Door> = HashMap::new();
        println!("{:?}, ANDANDAND {:?}", lock_pos, door_pos);
        if (lock_pos.len() == 0) || (door_pos.len() == 0) {
            ()
        } else {
            for (id, index) in ids {
                doors.entry(id as u32).or_insert_with(|| Door {
                    lock_for_door: lock_pos[index] - door_pos[index],
                    lock_pos: lock_pos[index],
                    scale: scales[index],
                });
            }
        }

        let ser_matrices = matrices
            .iter()
            .map(|m| m.matrix)
            .collect::<Vec<[[f32; 4]; 4]>>();
        let ser_doors: HashMap<u32, BakedDoor> = doors
            .iter()
            .map(|(k, d)| {
                (
                    *k,
                    BakedDoor {
                        lock_for_door: d.lock_for_door.into(),
                        lock_pos: d.lock_pos.into(),
                        scale: d.scale.into(),
                    },
                )
            })
            .collect();
        let serialized_data = bincode::serialize::<BakedMeshes>(&BakedMeshes {
            vertex: all_verticles,
            index: all_indices,
            primitives: pris.clone(),
            baked_texture,
            matrices: ser_matrices,
            doors: ser_doors,
        })
        .expect("cant ser data");
        fs::write(reload_path.as_str(), serialized_data).expect("can write file");

        scene_helper::ram();
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