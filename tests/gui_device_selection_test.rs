#![cfg(feature = "gui")]

use qwen3tts::gui::{BackendKind, DevicePreference, DeviceSelection, WorkerEvent};
use std::sync::mpsc;
use std::time::{Duration, Instant};

fn wait_for_probe(selection: &mut DeviceSelection) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while selection.detected().is_none() {
        selection.poll();
        assert!(Instant::now() < deadline, "probe did not complete");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn probe_is_background_and_preserves_fallback_reason() {
    let caller = std::thread::current().id();
    let mut selection = DeviceSelection::default();
    assert_eq!(selection.preference(), DevicePreference::Auto);
    selection.request_probe_with(move |preference| {
        assert_ne!(std::thread::current().id(), caller);
        assert_eq!(preference, DevicePreference::Auto);
        Ok("CPU：CUDA driver unavailable".into())
    });
    wait_for_probe(&mut selection);
    assert!(
        selection
            .detected()
            .unwrap()
            .as_ref()
            .unwrap()
            .contains("driver unavailable")
    );
    assert!(selection.actual().is_none());
}

#[test]
fn old_probe_cannot_overwrite_new_preference() {
    let (tx, rx) = mpsc::channel();
    let mut selection = DeviceSelection::default();
    selection.request_probe_with(move |_| {
        rx.recv().unwrap();
        Ok("old CUDA".into())
    });
    selection.set_preference(DevicePreference::Cpu);
    selection.request_probe_with(|preference| {
        assert_eq!(preference, DevicePreference::Cpu);
        Ok("new CPU".into())
    });
    wait_for_probe(&mut selection);
    tx.send(()).unwrap();
    selection.poll();
    assert_eq!(selection.detected(), Some(&Ok("new CPU".into())));
}

#[test]
fn worker_resolution_updates_actual_without_rewriting_probe() {
    let mut selection = DeviceSelection::default();
    selection.request_probe_with(|_| Ok("CUDA available".into()));
    wait_for_probe(&mut selection);
    selection.handle_event(&WorkerEvent::DeviceResolved(Ok("CPU fallback".into())));
    assert_eq!(selection.actual(), Some(&Ok("CPU fallback".into())));
    assert_eq!(selection.detected(), Some(&Ok("CUDA available".into())));
    selection.handle_event(&WorkerEvent::DeviceResolved(Err("GPU unavailable".into())));
    assert_eq!(selection.actual(), Some(&Err("GPU unavailable".into())));
}

#[test]
fn device_controls_render_in_headless_egui() {
    let ctx = egui::Context::default();
    let mut selection = DeviceSelection::default();
    selection.request_probe_with(|_| Ok("CPU test device".into()));
    wait_for_probe(&mut selection);
    let output = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            selection.show(ui, BackendKind::Python, false);
        });
    });
    assert!(!output.shapes.is_empty());
    let rendered = format!("{:?}", output.shapes);
    assert!(rendered.contains("Python 文字前端的 GPU"));
    assert!(rendered.contains("CPU test device"));
    assert_eq!(selection.preference(), DevicePreference::Auto);
}

#[test]
fn real_runtime_auto_and_forced_cpu_probe() {
    let mut selection = DeviceSelection::default();
    selection.request_probe();
    wait_for_probe(&mut selection);
    println!("REAL_AUTO_DEVICE={:?}", selection.detected());
    assert!(selection.detected().unwrap().is_ok());
    selection.set_preference(DevicePreference::Cpu);
    selection.request_probe();
    wait_for_probe(&mut selection);
    let name = selection.detected().unwrap().as_ref().unwrap();
    assert!(name.contains("CPU"));
    println!("REAL_FORCED_CPU_DEVICE={name}");
}

#[cfg(feature = "candle-llm")]
#[test]
fn native_device_controls_render_without_python_scope_notice() {
    let ctx = egui::Context::default();
    let mut selection = DeviceSelection::default();
    selection.request_probe_with(|_| Ok("CUDA test device".into()));
    wait_for_probe(&mut selection);
    selection.handle_event(&WorkerEvent::DeviceResolved(Ok("CPU actual".into())));
    let output = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            selection.show(ui, BackendKind::Candle, true);
        });
    });
    let rendered = format!("{:?}", output.shapes);
    assert!(rendered.contains("CUDA test device"));
    assert!(rendered.contains("CPU actual"));
    assert!(!rendered.contains("Python 文字前端的 GPU"));
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cpu_build_auto_works_and_forced_cuda_reports_build_limit() {
    let mut selection = DeviceSelection::default();
    selection.request_probe();
    wait_for_probe(&mut selection);
    assert!(
        selection
            .detected()
            .unwrap()
            .as_ref()
            .unwrap()
            .contains("未編譯 CUDA")
    );
    selection.set_preference(DevicePreference::Cuda);
    selection.request_probe();
    wait_for_probe(&mut selection);
    assert!(
        selection
            .detected()
            .unwrap()
            .as_ref()
            .unwrap_err()
            .contains("未編譯 CUDA")
    );
    selection.set_preference(DevicePreference::Cpu);
    selection.request_probe();
    wait_for_probe(&mut selection);
    assert!(selection.detected().unwrap().is_ok());
}
