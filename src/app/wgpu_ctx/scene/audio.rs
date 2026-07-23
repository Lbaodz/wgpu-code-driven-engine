use rodio::{Decoder, DeviceSinkBuilder, MixerDeviceSink, Player};
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

pub struct Audio {
    device: MixerDeviceSink,
    cache: HashMap<String, Arc<[u8]>>,
    active: HashMap<String, (Player, IsPlaying)>,
    pub volume: f32,
}

struct IsPlaying {
    is_playing: bool,
}

impl Audio {
    pub fn new() -> Self {
        Self {
            device: DeviceSinkBuilder::open_default_sink().expect("no audio"),
            cache: HashMap::new(),
            active: HashMap::new(),
            volume: 50.0,
        }
    }

    pub fn load(&mut self, name: &str, path: &str) {
        let bytes = std::fs::read(path).expect("no path found");
        self.cache.insert(name.to_string(), Arc::from(bytes));
    }

    pub fn play(&self, name: &str, speed: f32, volume: f32) {
        let volume = volume * self.volume / 100.0;
        let bytes = self.cache.get(&name.to_string()).expect("no byte");
        let cursor = Cursor::new(bytes.clone());
        let source = Decoder::new(cursor).expect("no cursor");
        let player = Player::connect_new(&self.device.mixer());
        player.set_speed(speed);
        player.set_volume(volume);
        player.append(source);
        player.detach();
    }

    pub fn stop(&mut self, thread: &str) {
        self.active.remove(&thread.to_string());
    }

    pub fn stop_slowly(&mut self, thread: &str, speed_fading: f32, dt: f32) {
        let Some((player, playing)) = self.active.get_mut(&thread.to_string()) else {
            println!("KEY: {:?}", self.active.keys());
            return;
        };
        playing.is_playing = false;
        if player.volume() > 0.01 {
            let dt_fade = speed_fading * dt;
            let new_volume = player.volume() - dt_fade * player.volume();
            player.set_volume(new_volume);
        } else {
            self.active.remove(&thread.to_string());
        }
    }

    pub fn play_again(&mut self, thread: &str, name: &str, speed: f32, volume: f32) {
        self.stop(thread);
        let volume = volume * self.volume / 100.0;
        let bytes = self.cache.get(&name.to_string()).expect("no byte");
        let cursor = Cursor::new(Arc::from(bytes.clone()));
        let source = Decoder::new(cursor).expect("no decode");
        let player = Player::connect_new(self.device.mixer());
        player.set_speed(speed);
        player.set_volume(volume);
        player.append(source);
        self.active
            .insert(thread.to_string(), (player, IsPlaying { is_playing: true }));
    }

    pub fn is_playing(&self, thread: &str) -> bool {
        if let Some((player, playing)) = self.active.get(&thread.to_string()) {
            if player.empty() || !playing.is_playing {
                false
            } else {
                true
            }
        } else {
            false
        }
    }
}