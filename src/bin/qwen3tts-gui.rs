//! # Qwen3-TTS GUI — 圖形化語音合成應用程式
//!
//! 基於 egui/eframe 的桌面 GUI，支援文字轉語音合成與播放。
//!
//! ## 使用方式
//!
//! ```bash
//! # 基本啟動（Python 後端）
//! cargo run --bin qwen3tts-gui
//!
//! # 使用 Candle 原生後端
//! cargo run --bin qwen3tts-gui --features candle-llm
//!
//! # 使用 CUDA 加速
//! cargo run --bin qwen3tts-gui --features "candle-llm cuda"
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
        Box::new(|_cc| {
            _cc.egui_ctx.set_visuals(egui::Visuals::dark());
            _cc.egui_ctx.set_pixels_per_point(1.0);
            Ok(Box::new(qwen3tts::gui::TtsGuiApp::new()))
        }),
    )
}
