struct LightData {
    matrix: mat4x4f,
    color: vec4f,
    dir: vec4f,
    id: vec4f,
}

struct CacheLight {
    lights_count: u32,
    lights_in_tile: array<u32, 15>,
}

@group(0) @binding(0)
var<uniform> camera: mat4x4f;
@group(1) @binding(0)
var<storage, read> all_lights_data: array<LightData>;
@group(1) @binding(1)
var<storage, read_write> cache_light: array<CacheLight>;

@compute @workgroup_size(16, 16)
fn c_main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u,
) {
    
}