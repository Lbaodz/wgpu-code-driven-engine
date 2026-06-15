struct VxIn {
    @location(0) pos: vec3f,
    @location(1) color: vec3f,
}

struct FragOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
}

@vertex
fn vs_main(input: VxIn) -> FragOut {
    return FragOut(
        vec4f(input.pos, 1.0),
        vec4f(input.color, 1.0)
    );
}

@fragment
fn fs_main(in: FragOut) -> @location(0) vec4f {
    return in.color;
};