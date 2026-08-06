use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use pub_fields::pub_fields;
use serde::{Deserialize, Serialize};

#[pub_fields]
pub struct Camera {
    eye: Vec3,
    target: Vec3,
    up: Vec3,
    aspect: f32,
    fov: f32,
    near: f32,
    yaw: f32,
    pitch: f32,
    is_moving: bool,
    is_rotating: bool,
    planes: Option<Planes>,
}

#[pub_fields]
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UniformCamera {
    uniform: Mat4,
}

impl Camera {
    pub fn make_camera(&self) -> Mat4 {
        let view = glam::camera::rh::view::look_at_mat4(self.eye, self.target, self.up);
        let proj = glam::camera::rh::proj::directx::perspective_infinite_reverse(
            self.fov.to_radians(),
            self.aspect,
            self.near,
        );

        proj * view
    }

    pub fn update_target(&mut self) -> Vec3 {
        let rad_yaw = self.yaw.to_radians();
        let rad_pitch = self.pitch.to_radians();

        let new_vector = Vec3::new(
            rad_yaw.cos() * rad_pitch.cos(),
            rad_pitch.sin(),
            rad_yaw.sin() * rad_pitch.cos(),
        );
        self.target = self.eye + new_vector;
        new_vector
    }

    pub fn update_planes(&mut self, value: Planes) {
        if let Some(planes) = &mut self.planes {
            *planes = value;
        }
    }

    pub fn take_plane(&self) -> Planes {
        if let Some(planes) = &self.planes {
            *planes
        } else {
            Planes::build_plane_from_matrix4(self.make_camera())
        }
    }
}

#[pub_fields]
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
struct Plane {
    normal: [f32; 3],
    d: f32,
}

#[pub_fields]
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct Planes {
    planes: [Plane; 6],
}

impl Planes {
    pub fn build_plane_from_matrix4(matrix4: Mat4) -> Self {
        let m = matrix4;
        let (x, y, z, w) = (m.row(0), m.row(1), m.row(2), m.row(3));
        let make_plane = |nx, ny, nz, d| {
            let length = Vec3::new(nx, ny, nz).length();
            Plane {
                normal: [nx / length, ny / length, nz / length],
                d: d / length,
            }
        };
        Self {
            planes: [
                make_plane(w.x + x.x, w.y + x.y, w.z + x.z, w.w + x.w),
                make_plane(w.x - x.x, w.y - x.y, w.z - x.z, w.w - x.w),
                make_plane(w.x + y.x, w.y + y.y, w.z + y.z, w.w + y.w),
                make_plane(w.x - y.x, w.y - y.y, w.z - y.z, w.w - y.w),
                make_plane(z.x, z.y, z.z, z.w),
                make_plane(w.x - z.x, w.y - z.y, w.z - z.z, w.w - z.w),
            ],
        }
    }

    pub fn frustum_culling(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        let box_min = Vec3::from_array(min);
        let box_max = Vec3::from_array(max);
        for plane in &self.planes {
            let normal = Vec3::from_array(plane.normal);
            let is_positive = Vec3::ZERO.cmple(normal);

            let positive_normal = Vec3::select(is_positive, box_max, box_min);

            if normal.dot(positive_normal) + plane.d < 0.0 {
                return false;
            }
        }
        true
    }
}

#[pub_fields]
pub struct LightCtx {
    light_bg: wgpu::BindGroup,
    light_views: Vec<wgpu::TextureView>,
    depth_view: wgpu::TextureView,
    shadow_bg: wgpu::BindGroup,
    light_pipeline: wgpu::RenderPipeline,
    compute_lights_bg: wgpu::BindGroup,
}

impl LightCtx {
    pub fn new(
        light_bg: wgpu::BindGroup,
        light_views: Vec<wgpu::TextureView>,
        shadow_bg: wgpu::BindGroup,
        depth_view: wgpu::TextureView,
        light_pipeline: wgpu::RenderPipeline,
        compute_lights_bg: wgpu::BindGroup,
    ) -> Self {
        Self {
            light_bg,
            light_views,
            shadow_bg,
            depth_view,
            light_pipeline,
            compute_lights_bg,
        }
    }
}

#[pub_fields]
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable, Serialize, Deserialize)] // BAKE LATER
pub struct LightData {
    light_matrices: Mat4,
    color: [f32; 4], // index 3 is type id
    dir: [f32; 4],   // index 3 is density
    id: [f32; 4],    // id light[0] and range[1]
}

#[pub_fields]
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LightDataAlign {
    data: LightData,
    _pad: [u32; 36],
}

#[pub_fields]
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct Light {
    data: LightData,
    planes: Planes,
}

impl Light {
    pub fn new_spot_light(
        pos: Vec3,
        dir: Vec3,
        fov: f32,
        near: f32,
        density: f32,
        mut color: [f32; 4],
        range: f32,
    ) -> Self {
        let dir_n = dir.normalize();
        let mut up = Vec3::Y;
        if dir_n.y.abs() > 0.99 {
            up = Vec3::Z
        };
        let view = glam::camera::rh::view::look_at_mat4(pos, dir_n, up);
        let proj = glam::camera::rh::proj::directx::perspective_infinite_reverse(
            fov.to_radians(),
            1.0,
            near,
        );
        color[3] = 1.0;
        let mut id = [0.0; 4];
        id[1] = range;

        let matrix = proj * view;
        Self {
            data: LightData {
                light_matrices: matrix,
                dir: [dir_n.x, dir_n.y, dir_n.z, density],
                color,
                id,
            },
            planes: Planes::build_plane_from_matrix4(matrix),
        }
    }
}

// helper
#[macro_export]
macro_rules! v3 {
    ($x:expr, $y:expr, $z:expr $(,)?) => {
        Vec3::new($x, $y, $z)
    };
}
