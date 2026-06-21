@group(0) @binding(0)
var<uniform> camera: mat4x4<f32>;

struct VxIn {
    @location(0) pos: vec3f,
    @location(1) color: vec3f,
    @location(2) normal: vec3f,
}

struct FragOut {
    @builtin(position) pos: vec4f,
    @location(3) light_pos: vec4f,
    @location(0) color: vec3f,
    @location(1) normal: vec3f,
}

@vertex
fn vs_main(input: VxIn) -> FragOut {
    return FragOut(
        camera * vec4<f32>(input.pos, 1.0),
        camera * vec4<f32>(3.0, -3.0, 1.0, 1.0),
        vec3<f32>(input.color),
        vec3<f32>(input.normal)
    );
};

@fragment
fn fs_main(in: FragOut) -> @location(0) vec4f {
    let ambient_strength: f32 = 0.1;
    let light_color = vec3f(1.0, 0.0, 1.0);
    let ambient_color: vec3f = light_color * ambient_strength;
    let light_dir: vec3f = normalize(in.light_pos.xyz - in.pos.xyz);
    let diff: f32 = max(dot(in.normal, light_dir), 0.0);
    let diff_color: vec3f = diff * in.color;

    let final_light: vec3f = ambient_color + diff_color;
    let final_color: vec3f = in.color.rgb * final_light;
    return vec4f(final_color, 1.0);
};