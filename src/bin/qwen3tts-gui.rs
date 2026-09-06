//! # Qwen3-TTS GUI — 圖形化語音合成應用程式
//!
//! 基於 egui/eframe 的桌面 GUI，支援文字轉語音合成與播放。
//!
//! ## 使用方式
//!
//! ```bash
//! # CPU + Candle 原生後端
//! cargo run --bin qwen3tts-gui --no-default-features \
//!     --features "cpu,candle-llm,gui"
//!
//! # Python bridge（不啟用 Candle Talker）
//! cargo run --bin qwen3tts-gui --no-default-features --features "cpu,gui"
//!
//! # NVIDIA CUDA 加速
//! cargo run --bin qwen3tts-gui --no-default-features \
//!     --features "cpu,candle-llm,gui,cuda"
//! ```

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([820.0, 720.0])
            .with_min_inner_size([600.0, 500.0])
            .with_title("Qwen3-TTS 語音合成"),
        ..Default::default()
    };

    eframe::run_native(
        "Qwen3-TTS 語音合成",
        options,
        Box::new(|cc| {
            qwen3tts::gui::install_cjk_fonts(&cc.egui_ctx);
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            cc.egui_ctx.set_pixels_per_point(1.0);
            Ok(Box::new(qwen3tts::gui::TtsGuiApp::new()))
        }),
    )
}
