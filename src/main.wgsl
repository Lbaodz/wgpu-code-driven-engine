struct LightData {
    matrix: mat4x4f,
    color: vec4f,
    dir: vec4f,
}

@group(0) @binding(0)
var<uniform> camera: mat4x4<f32>;
@group(1) @binding(0)
var t: texture_2d<f32>;
@group(1) @binding(1)
var s: sampler;
@group(2) @binding(0)
var<uniform> model_matrix: mat4x4<f32>;

@group(3) @binding(0)
var<storage, read> all_lights: array<LightData>;
@group(3) @binding(1)
var all_tt: texture_depth_2d_array;
@group(3) @binding(2)
var sample: sampler_comparison;

struct VxIn {
    @location(0) pos: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
}

struct FragOut {
    @builtin(position) pos: vec4f,
    @location(2) world_pos: vec4f,
    @location(3) camera: vec4f,
    @location(0) uv: vec2f,
    @location(1) normal: vec3f,
}

fn inverse_mat3(m: mat3x3<f32>) -> mat3x3<f32> {
    let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2];
    let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2];
    let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2];

    let b01 = a22 * a11 - a12 * a21;
    let b11 = -a22 * a10 + a12 * a20;
    let b21 = a21 * a10 - a11 * a20;

    let det = a00 * b01 + a01 * b11 + a02 * b21;

    if (abs(det) < 1e-6) {
        return mat3x3<f32>(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
    }

    let inv_det = 1.0 / det;
    return mat3x3<f32>(
        vec3f(b01, (-a22 * a01 + a02 * a21), (a12 * a01 - a02 * a11)) * inv_det,
        vec3f(b11, (a22 * a00 - a02 * a20), (-a12 * a00 + a02 * a10)) * inv_det,
        vec3f(b21, (-a21 * a00 + a01 * a20), (a11 * a00 - a01 * a10)) * inv_det
    );
}

@vertex
fn vs_main(input: VxIn) -> FragOut {
    var out: FragOut;
    let world_position = model_matrix * vec4<f32>(input.pos, 1.0);
    let model_3x3 = mat3x3f(
        model_matrix[0].xyz,
        model_matrix[1].xyz,
        model_matrix[2].xyz,
    );
    let normal_mat = transpose(inverse_mat3(model_3x3));
    let normal = normalize(normal_mat * input.normal);
    out.pos = camera * world_position;
    out.world_pos = world_position;
    out.camera = camera * vec4<f32>(1.0, 1.0, 1.0, 1.0);
    out.uv = vec2<f32>(input.uv);
    out.normal = normal;
    return out;
};

@fragment
fn fs_main(in: FragOut) -> @location(0) vec4f {
    let ambient_light = vec3f(0.4);
    let shadow_factor = calculate_shadow(in);
    let final_shadow = ambient_light + shadow_factor;
    let tt = textureSample(t, s, in.uv);
    //return vec4f(l_uv.x, l_uv.y, d, 1.0);
    return tt * vec4f(final_shadow, 1.0);
    // * vec4f(sin(tt.x + in.pos.x) * cos(in.camera.x), sin(in.camera.b) * cos(tt.b + tt.r), cos(in.camera.x) * sin(tt.w + tt.z), cos(tt.y + tt.b + tt.g) * sin(in.world_pos.b));
    /* return vec4<f32>(sin(tt.x + in.pos.x) * cos(in.camera.x),
    sin(tt.y + in.pos.y) * sin(in.camera.y),
    sin(tt.x + tt.y * in.pos.z) * cos(in.camera.x), 1.0); */
};

@vertex
fn vs_transparency_main(input: VxIn) -> FragOut {
    var out: FragOut;
    let world_position = model_matrix * vec4<f32>(input.pos, 1.0);
    let model_3x3 = mat3x3f(
        model_matrix[0].xyz,
        model_matrix[1].xyz,
        model_matrix[2].xyz,
    );
    let normal_mat = transpose(inverse_mat3(model_3x3));
    let normal = normalize(normal_mat * input.normal);
    out.pos = camera * world_position;
    out.world_pos = world_position;
    out.camera = camera * vec4<f32>(1.0, 1.0, 1.0, 1.0);
    out.uv = vec2<f32>(input.uv);
    out.normal = normal;
    return out;
}

@fragment
fn fs_transparency_main(in: FragOut) -> @location(0) vec4f {
    let ambient_light = vec3f(0.4);
    let shadow_factor = calculate_shadow_transparency(in);
    let final_shadow = ambient_light + shadow_factor;
    //return vec4f(l_uv.x, l_uv.y, d, 1.0);
    //return vec4f(1.0, 0.0, 0.0, 0.6);
    return textureSample(t, s, in.uv) * vec4f(final_shadow, 0.75);
    /*let tt = textureSample(t, s, in.uv);
    return vec4<f32>(sin(tt.x + in.pos.x) * cos(in.camera.x),
    sin(tt.y + in.pos.y) * sin(in.camera.y),
    sin(tt.x + tt.y * in.pos.z) * cos(in.camera.x), 1.0); */
};

const POISSON_DISK = array<vec2f, 8>(
    vec2f(-0.94201624, -0.39906216),
    vec2f(0.94558609, -0.76890725),
    vec2f(-0.094184101, -0.92938870),
    vec2f(0.34495938, 0.29387760),
    vec2f(-0.91588581, 0.45771432),
    vec2f(-0.33651259, 0.75893221),
    vec2f(0.35951867, -0.21198902),
    vec2f(0.79115914, 0.19090169)
);

fn calculate_shadow_transparency(in: FragOut) -> vec3f {
    let normal = in.normal;
    var shadow_factor = vec3f(0.0);

    for (var i: u32 = 0u; i < arrayLength(&all_lights); i = i + 1u) {
        let light = all_lights[i];
        let dir = light.dir.xyz;
        let dot_val = dot(-dir, normal);
        let light_density = light.dir.w;
        let light_color = light.color.xyz;

        if (dot_val <= 1e-6 || light.dir.w <= 0.0001) {
            continue;
        }

        let light_view: vec4f = light.matrix * in.world_pos;
        let proj_light_view = light_view.xyz / light_view.w;
        let light_uv = vec2f(
            proj_light_view.x * 0.5 + 0.5,
            -proj_light_view.y * 0.5 + 0.5
        );
        let d = proj_light_view.z;

        if (light_uv.x < 0.0 || light_uv.x > 1.0 || 
            light_uv.y < 0.0 || light_uv.y > 1.0 || 
            d < 0.0 || d > 1.0) {
            continue;
        }
        let base_bias = 0.005;
        let light_visibility = textureSampleCompare(
            all_tt, sample, light_uv, i, d - base_bias
        );

        shadow_factor += light_density * light_visibility * light_color * dot_val;
    }

    return shadow_factor;
}

fn calculate_shadow(in: FragOut) -> vec3f {
    let normal = in.normal;
    var shadow_factor = vec3f(0.0);
    let inv_tex_size = vec2f(1.0) / vec2f(1024.0);
    
    let num_lights = arrayLength(&all_lights); 

    for (var i: u32 = 0u; i < num_lights; i = i + 1u) {
        let light = all_lights[i];
        let dir = light.dir.xyz;
        let dot_val = dot(-dir, normal);
        let light_density = light.dir.w;
        let light_color = light.color.xyz;

        if (dot_val <= 1e-6 || light.dir.w <= 0.0001) { continue; }

        let light_view: vec4f = light.matrix * in.world_pos;
        let proj_light_view = light_view.xyz / light_view.w;
        let light_uv = vec2f(proj_light_view.x * 0.5 + 0.5, -proj_light_view.y * 0.5 + 0.5);
        let d = proj_light_view.z;

        if (light_uv.x < 0.0 || light_uv.x > 1.0 || light_uv.y < 0.0 || light_uv.y > 1.0 || d < 0.0 || d > 1.0) { continue; }

        let base_bias = 0.005;
        
        let filter_scale = inv_tex_size; 

        var visibility = 0.0;
        
        visibility += textureSampleCompare(all_tt, sample, light_uv, i, d - base_bias);
         /*visibility += textureSampleCompare(all_tt, sample, light_uv + POISSON_DISK[1] * filter_scale, i, d - base_bias);
        visibility += textureSampleCompare(all_tt, sample, light_uv + POISSON_DISK[2] * filter_scale, i, d - base_bias);
        visibility += textureSampleCompare(all_tt, sample, light_uv + POISSON_DISK[3] * filter_scale, i, d - base_bias);
        visibility += textureSampleCompare(all_tt, sample, light_uv + POISSON_DISK[4] * filter_scale, i, d - base_bias);
        visibility += textureSampleCompare(all_tt, sample, light_uv + POISSON_DISK[5] * filter_scale, i, d - base_bias);
        visibility += textureSampleCompare(all_tt, sample, light_uv + POISSON_DISK[6] * filter_scale, i, d - base_bias);
        visibility += textureSampleCompare(all_tt, sample, light_uv + POISSON_DISK[7] * filter_scale, i, d - base_bias);*/
        
        visibility /= 1.0;
        shadow_factor += light_density * visibility * light_color * dot_val;
    }
    return shadow_factor;
}
