// debug
use sysinfo::{Pid, System};
pub fn ram() {
    let mut sys = System::new_all();
    sys.refresh_all();
    if let Some(process) = sys.process(Pid::from(std::process::id() as usize)) {
        println!("RAM: {} KB", process.memory() / 1024);
    }
}