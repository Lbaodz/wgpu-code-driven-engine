// debug
use sysinfo::{Pid, System};
pub fn ram(label: &str) {
    let mut sys = System::new_all();
    sys.refresh_all();
    if let Some(process) = sys.process(Pid::from(std::process::id() as usize)) {
        let ram_kb = process.memory() / 1024;
        let ram_mb = ram_kb as f32 / 1024.0;
        let ram_gb = ram_mb / 1024.0;
        println!(
            "{label} has: RAM: {} KB OR: {:.2} MB OR {:.3} GB",
            ram_kb, ram_mb, ram_gb
        );
    }
}
