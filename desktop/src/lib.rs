//! NoMusic desktop: the engine and live audio plumbing shared by the tray app (NoMusic.exe) and the
//! command-line tool (nomusic-cli.exe).

pub mod default_device;
pub mod engine;
pub mod live;

use engine::{CHANNELS, SAMPLE_RATE};

/// Writes interleaved 48 kHz stereo as 16-bit PCM.
pub fn write_wav(path: &str, samples: &[f32]) -> Result<(), String> {
    let spec = hound::WavSpec {
        channels: CHANNELS as u16,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(|e| format!("{path}: {e}"))?;
    for &s in samples {
        writer
            .write_sample((s.clamp(-1.0, 1.0) * 32767.0).round() as i16)
            .map_err(|e| format!("{path}: {e}"))?;
    }
    writer.finalize().map_err(|e| format!("{path}: {e}"))
}
