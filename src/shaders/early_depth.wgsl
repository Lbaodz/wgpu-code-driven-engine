@group(0) @binding(0)
var<uniform> camera: mat4x4<f32>;
@group(1) @binding(0)
var<uniform> model_matrix: mat4x4<f32>;

@vertex
fn vs_z_main(@location(0) pos: vec3f) -> @builtin(position) vec4f {
    let world_position = model_matrix * vec4<f32>(pos, 1.0);
    return camera * world_position;
};