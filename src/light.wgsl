struct LightData {
    matrix: mat4x4f,
    color: vec3f,
    dir: vec3f,
    density: f32,
}

struct VtIn {
    @location(0) vt: vec3f,
    @location(1) nor: vec3f,
    @location(2) uv: vec2f,
}

@group(0) @binding(0)
var<uniform> light_view: LightData;
@group(1) @binding(0)
var<uniform> model_mtx: mat4x4f;

@vertex
fn vs_main(vt_in: VtIn) -> @builtin(position) vec4f {
    return light_view.matrix * model_mtx * vec4f(vt_in.vt, 1.0);
}