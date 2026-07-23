use pub_fields::pub_fields;

pub enum GameLevel {
    Base,
}

#[pub_fields] 
#[derive(Default)]
pub struct PerformanceState {
    low: bool,
    mid: bool,
    ok: bool,
    high: bool,
    very_high: bool,
    epic: bool,
}

impl PerformanceState {
    pub fn all_false(&self) -> bool {
        ![
            self.low,
            self.mid,
            self.ok,
            self.high,
            self.very_high,
            self.epic,
        ]
        .into_iter()
        .any(|x| x)
    }
}

pub enum GameState {
    Menu,
    Play,
    Settings,
    Exit,
    Loading,
}

#[pub_fields] 
pub struct FileManager {
    model_paths: Vec<String>,
    audio_paths: Vec<String>,
}

#[pub_fields] 
#[derive(Default)]
pub struct InputState {
    w: bool,
    s: bool,
    a: bool,
    d: bool,
    q: bool,
    e: bool,
    shift: bool,
}