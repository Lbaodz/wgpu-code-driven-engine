use cgmath::{Point3, Vector3, Matrix4, Deg, InnerSpace, dot};
use pub_fields::pub_fields;

#[pub_fields] 
pub struct Camera {
    eye: Point3<f32>,
    target: Point3<f32>,
    up: Vector3<f32>,
    aspect: f32,
    fov: f32,
    near: f32,
    far: f32,
    yaw: f32,
    pitch: f32,
    is_moving: bool,
    is_rotating: bool,
}

#[pub_fields] 
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UniformCamera {
    uniform: [[f32; 4]; 4],
}


impl Camera {
    pub fn make_camera(&self) -> Matrix4<f32> {
        let view = Matrix4::look_at_rh(self.eye, self.target, self.up);
        let v_fov = 2.0 * ((self.fov.to_radians() / 2.0).tan() / self.aspect).atan();
        let proj = cgmath::perspective(Deg(v_fov.to_degrees()), self.aspect, self.near, self.far);
        let wgpu_matrix_correction = Matrix4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0,
        );
        wgpu_matrix_correction * proj * view
    }

    pub fn update_target(&mut self) -> Vector3<f32> {
        let rad_yaw = self.yaw.to_radians();
        let rad_pitch = self.pitch.to_radians();

        let new_vector = Vector3::new(
            rad_yaw.cos() * rad_pitch.cos(),
            rad_pitch.sin(),
            rad_yaw.sin() * rad_pitch.cos(),
        );
        self.target = self.eye + new_vector;
        new_vector
    }
}

#[pub_fields] 
#[derive(Default)]
struct Plane {
    normal: [f32; 3],
    d: f32,
}

#[pub_fields] 
pub struct Planes {
    planes: [Plane; 6],
}

impl Planes {
    pub fn build_plane_from_matrix4(matrix4: Matrix4<f32>) -> Self {
        let m = matrix4;
        let make_plane = |nx, ny, nz, d| {
            let length = Vector3::new(nx, ny, nz).magnitude();
            Plane {
                normal: [nx / length, ny / length, nz / length],
                d: d / length,
            }
        };
        Self {
            planes: [
                make_plane(m.x.w + m.x.x, m.y.w + m.y.x, m.z.w + m.z.x, m.w.w + m.w.x),
                make_plane(m.x.w - m.x.x, m.y.w - m.y.x, m.z.w - m.z.x, m.w.w - m.w.x),
                make_plane(m.x.w + m.x.y, m.y.w + m.y.y, m.z.w + m.z.y, m.w.w + m.w.y),
                make_plane(m.x.w - m.x.y, m.y.w - m.y.y, m.z.w - m.z.y, m.w.w - m.w.y),
                make_plane(m.x.w + m.x.z, m.y.w + m.y.z, m.z.w + m.z.z, m.w.w + m.w.z),
                make_plane(m.x.w - m.x.z, m.y.w - m.y.z, m.z.w - m.z.z, m.w.w - m.w.z),
            ],
        }
    }

    pub fn frustum_culling(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        for plane in &self.planes {
            let mut positive_normal: [f32; 3] = [0.0; 3];
            let normal = plane.normal;
            for i in 0..3 {
                if normal[i] >= 0.0 {
                    positive_normal[i] = max[i];
                } else {
                    positive_normal[i] = min[i];
                }
            }
            if dot::<Vector3<f32>>(normal.into(), positive_normal.into()) + plane.d < 0.0 {
                return false;
            }
        }
        true
    }
}