use std::path::Path;

use qwen3tts::{Decoder12Hz, DecoderConfig, TtsDecoder};

fn main() {
    let weight_dir = Path::new("weights/tokenizer");
    if !weight_dir.join("codebook.safetensors").exists() {
        eprintln!("weights/tokenizer/ 不存在，請先下載權重");
        std::process::exit(1);
    }

    let device = candle_core::Device::Cpu;
    let config = DecoderConfig::realtime();
    let mut decoder =
        Decoder12Hz::from_safetensors(config, weight_dir, &device).expect("載入權重失敗");

    let output_path = "output.wav";
    let sample_rate = 24000u32;
    let num_frames = 50;

    println!("正在解碼 {num_frames} 幀 (12Hz, 24kHz)…");
    let mut all_samples: Vec<f32> = Vec::with_capacity(num_frames * 1920);

    for i in 0..num_frames {
        let tokens: Vec<u16> = vec![
            (i as u16 * 42 + 1) % 2048,
            (i as u16 * 73 + 2) % 2048,
            (i as u16 * 13 + 3) % 2048,
            (i as u16 * 99 + 4) % 2048,
            (i as u16 * 55 + 5) % 2048,
            (i as u16 * 31 + 6) % 2048,
            (i as u16 * 87 + 7) % 2048,
            (i as u16 * 44 + 8) % 2048,
            (i as u16 * 66 + 9) % 2048,
            (i as u16 * 22 + 10) % 2048,
            (i as u16 * 11 + 11) % 2048,
            (i as u16 * 77 + 12) % 2048,
            (i as u16 * 88 + 13) % 2048,
            (i as u16 * 33 + 14) % 2048,
            (i as u16 * 50 + 15) % 2048,
            (i as u16 * 29 + 16) % 2048,
        ];

        let frame = decoder.decode_chunk(&tokens).expect("解碼幀失敗");
        all_samples.extend_from_slice(&frame);

        if i % 10 == 0 {
            println!("  幀 {}/{}", i + 1, num_frames);
        }
    }

    let duration_sec = all_samples.len() as f64 / sample_rate as f64;
    println!(
        "\n產出 {:.2} 秒音頻 ({} 個樣本)",
        duration_sec,
        all_samples.len()
    );

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(output_path, spec).expect("無法建立 WAV 檔案");

    for &sample in &all_samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let int_sample = (clamped * i16::MAX as f32) as i16;
        writer.write_sample(int_sample).expect("寫入樣本失敗");
    }

    writer.finalize().expect("關閉 WAV 檔案失敗");
    println!(
        "✅ 已儲存: {output_path} ({:.1} MB)",
        all_samples.len() as f64 * 2.0 / 1_048_576.0
    );
}
