@group(0) @binding(0)
var<uniform> camera: mat4x4<f32>;
@group(1) @binding(0)
var t: texture_2d<f32>;
@group(1) @binding(1)
var s: sampler;
@group(2) @binding(0)
var<uniform> model_matrix: mat4x4<f32>;

struct VxIn {
    @location(0) pos: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
}

struct FragOut {
    @builtin(position) pos: vec4f,
    // @location(3) light_pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) normal: vec3f,
}

@vertex
fn vs_main(input: VxIn) -> FragOut {
    return FragOut(
        camera * model_matrix * vec4<f32>(input.pos, 1.0),
        // camera * vec4<f32>(3.0, -3.0, 1.0, 1.0),
        vec2<f32>(input.uv),
        vec3<f32>(input.normal)
    );
};

@fragment
fn fs_main(in: FragOut) -> @location(0) vec4f {
    return textureSample(t, s, in.uv);
    //return vec4<f32>(in.uv.x, in.uv.y, in.uv.x * in.uv.y, 1.0);
};