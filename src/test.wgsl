struct FragOut { 
    @location(0) color: vec4f,
}

@vertex
fn vs_main(color: FragOut) {

}

@fragment
fn fs_main(color: FragOut) -> @location(0) color: ver4f {
    color.color
}