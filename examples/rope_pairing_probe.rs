//! # RoPE Interleaved 配對探測腳本
//!
//! 目的：觀察 `rope_interleaved=true/false` 兩種模式下，
//! 3D Multimodal RoPE 如何把維度分配到三個軸（text/audio/vision）。
//!
//! 依據 WO-8 規格：
//! - 不修改主線程式碼（`primitives.rs` / `config.rs`）
//! - 只做驗證與記錄
//!
//! 用法：
//!   cargo run --example rope_pairing_probe
//!
//! 輸出說明：
//!   對 head_dim=128 （mrope_section=[24,20,20]），列出每個維度 (0..127)
//!   所屬的軸（0=text, 1=audio, 2=vision）及配對維度。

use std::collections::BTreeMap;

/// 模擬 `mrope_axis_for_dim`（`rope_interleaved=true` 分支）
fn axis_interleaved(dim: usize, head_dim: usize, mrope_section: &[usize]) -> usize {
    let modality_num = mrope_section.len();
    let half_dim = head_dim / 2; // = 64
    // 計算 (dim % half_dim) 而非全維度（mirror 第一半）
    let axis_dim = dim % half_dim;
    // axis 1..modality_num-1 對應非文字模態（audio, vision）
    for axis in 1..modality_num {
        let start = axis;
        let end = mrope_section[axis] * modality_num;
        if axis_dim >= start && axis_dim < end && (axis_dim - start) % modality_num == 0 {
            return axis;
        }
    }
    // 不屬於軸 1/2 的維度全部歸軸 0（text）
    0
}

/// 模擬 `mrope_axis_for_dim`（`rope_interleaved=false` 分支）
fn axis_blocked(dim: usize, _head_dim: usize, mrope_section: &[usize]) -> usize {
    let modality_num = mrope_section.len();
    let mut offset = 0usize;
    // 每個模態會出現 2 次（因為 head_dim = 2 * sum(mrope_section)）
    for chunk in 0..(modality_num * 2) {
        let axis = chunk % modality_num;
        let section = mrope_section[axis];
        if dim < offset + section {
            return axis;
        }
        offset += section;
    }
    0
}

fn main() {
    let head_dim = 128usize;
    let mrope_section = vec![24usize, 20, 20];
    let modality_names = ["text (T)", "audio (A)", "vision (V)"];
    let sep = "=".repeat(80);

    println!("{sep}");
    println!("RoPE Interleaved 配對探測報告");
    println!("{sep}");
    println!("head_dim       = {head_dim}");
    println!("mrope_section  = {mrope_section:?}  (pairs: text=24, audio=20, vision=20)");
    println!(
        "總 pairs       = {} (= {})",
        mrope_section.iter().sum::<usize>(),
        head_dim / 2
    );
    println!();

    // ── Step 1: 列出每個維度的 axis 分配 ──
    let mut interleaved_axes: Vec<usize> = Vec::with_capacity(head_dim);
    let mut blocked_axes: Vec<usize> = Vec::with_capacity(head_dim);

    for d in 0..head_dim {
        interleaved_axes.push(axis_interleaved(d, head_dim, &mrope_section));
        blocked_axes.push(axis_blocked(d, head_dim, &mrope_section));
    }

    println!("─ STEP 1: 逐維度 axis 分配 ─");
    println!();
    println!("  維度範圍    | Interleaved (true)      | Blocked (false)");
    for chunk in 0..8 {
        let start = chunk * 16;
        let end = start + 15;
        let inter_slice: Vec<String> = interleaved_axes[start..=end]
            .iter()
            .map(|a| modality_names[*a].chars().next().unwrap().to_string())
            .collect();
        let block_slice: Vec<String> = blocked_axes[start..=end]
            .iter()
            .map(|a| modality_names[*a].chars().next().unwrap().to_string())
            .collect();
        println!(
            "  [{:>3}..{:>3}] | {} | {}",
            start,
            end,
            inter_slice.join(" "),
            block_slice.join(" ")
        );
    }

    // ── Step 2: 配對規則 ──
    println!();
    println!("─ STEP 2: RoPE 配對規則 ─");
    println!();
    println!("在 rotate_half_and_apply 中，維度 d (0..head_dim/2-1) 與 d+head_dim/2 配對。");
    println!("兩者所屬的 axis 必須相同角速度才能保證正確旋轉。");
    println!();

    let mut interleaved_mismatch = 0;
    let mut blocked_mismatch = 0;
    let half = head_dim / 2;
    for pair_idx in 0..half {
        let d1 = pair_idx;
        let d2 = pair_idx + half;
        if interleaved_axes[d1] != interleaved_axes[d2] {
            interleaved_mismatch += 1;
            if interleaved_mismatch <= 5 {
                println!(
                    "  ⚠️  Interleaved: 維度 {d1}(axis {}) 與 {d2}(axis {}) 軸不一致!",
                    interleaved_axes[d1], interleaved_axes[d2]
                );
            }
        }
        if blocked_axes[d1] != blocked_axes[d2] {
            blocked_mismatch += 1;
            if blocked_mismatch <= 5 {
                println!(
                    "  ⚠️  Blocked: 維度 {d1}(axis {}) 與 {d2}(axis {}) 軸不一致!",
                    blocked_axes[d1], blocked_axes[d2]
                );
            }
        }
    }
    if interleaved_mismatch == 0 {
        println!("  ✅ Interleaved: 所有配對維度 axis 一致");
    } else {
        println!("  ❌ Interleaved: {interleaved_mismatch} 對軸不一致");
    }
    if blocked_mismatch == 0 {
        println!("  ✅ Blocked: 所有配對維度 axis 一致");
    } else {
        println!("  ❌ Blocked: {blocked_mismatch} 對軸不一致");
    }

    // ── Step 3: 每個軸的維度密度圖 ──
    println!();
    println!("─ STEP 3: 每個軸的分布 — Interleaved ─");
    for axis in 0..modality_names.len() {
        let dims: Vec<usize> = interleaved_axes
            .iter()
            .enumerate()
            .filter(|&(_, &a)| a == axis)
            .map(|(i, _)| i)
            .collect();
        let count = dims.len();
        let expected = if axis == 0 {
            48 // (24 pairs + 4 residue in 60-63)  Wait, let me compute:
        // First half: 24 values for axis 0, but only 20 of them in the interleaved pattern (0, 3, 6, ... 57)
        // Plus 4 extra (60, 61, 62, 63) = 24 in first half
        // Total: 24 + 24 = 48 for axis 0
        } else {
            40 // 20 in first half + 20 in second half
        };
        println!(
            "  軸 {} ({:>12}): {:>2} 維度 {expected:>2} expected, dims={:?}",
            axis,
            modality_names[axis],
            count,
            &dims[..dims.len().min(12)]
        );
        if dims.len() > 12 {
            println!("    ... 以及另外 {} 個維度", dims.len() - 12);
        }
    }

    println!();
    println!("─ STEP 3b: 每個軸的分布 — Blocked ─");
    for axis in 0..modality_names.len() {
        let dims: Vec<usize> = blocked_axes
            .iter()
            .enumerate()
            .filter(|&(_, &a)| a == axis)
            .map(|(i, _)| i)
            .collect();
        let count = dims.len();
        println!(
            "  軸 {} ({:>12}): {:>2} 維度, dims={:?}",
            axis,
            modality_names[axis],
            count,
            &dims[..dims.len().min(12)]
        );
        if dims.len() > 12 {
            println!("    ... 以及另外 {} 個維度", dims.len() - 12);
        }
    }

    // ── Step 4: 量化差異 ──
    println!();
    println!("─ STEP 4: 差異摘要 ─");
    let diff_count = interleaved_axes
        .iter()
        .zip(blocked_axes.iter())
        .filter(|(a, b)| a != b)
        .count();
    println!("  兩個模式在 {diff_count}/{head_dim} 個維度上 axis 分配不同");

    // 哪個維度不同？
    let diffs: Vec<usize> = interleaved_axes
        .iter()
        .zip(blocked_axes.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    println!("  不同的維度: {diffs:?}");

    // ── Step 5: 與 qwentts.cpp GGUF metadata 比對 ──
    println!();
    println!("─ STEP 5: 與 qwentts.cpp GGUF metadata 關係 ─");
    println!("  qwentts.cpp GGUF metadata 記錄 `mrope_interleaved=false`");
    println!("  Rust config.rs 預設 `rope_interleaved=true`");
    println!();
    println!("  現有 Rust 驗證（talker_single_frame / two_frame fixture）使用");
    println!("  default config（rope_interleaved=true），且已通過 exact token match");
    println!("  這不代表某個模式「正確」或「錯誤」，只說明：");
    println!("  - 在目前的測試 fixture 覆蓋範圍內，兩種模式可能產生相同或不同的輸出。");
    println!("  如果 token match 驗證使用了 rope_interleaved=true 的 config 產生預期輸出，");
    println!("  那 true 就是與 PyTorch 參考一致的模式。反之亦然。");
    println!();

    // ── Step 6: 三頻率軸分離示意 ──
    println!("─ STEP 6: 軸分離原理 ─");
    println!("  3D Multimodal RoPE 把 head_dim 切成三個區段（text/audio/vision），");
    println!("  每個區段使用自己的位置 ID 計算旋轉角度，");
    println!("  使得不同模態的 QK 內積在同一維度可以對齊不同位置。");
    println!("  rope_interleaved 只影響如何將維度分配到三個區段：");
    println!("  - true:  三個區段的維度交錯排列（每 3 個維度一組分配）");
    println!("  - false: 每個區段的維度連續排列（blocked）");
    println!();

    // ── 總結表 ──
    println!("─ 配對規則總結 ─");
    println!();
    let mut summary: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for d in 0..half {
        let pair1 = d;
        let pair2 = d + half;
        let key = (interleaved_axes[pair1], interleaved_axes[pair2]);
        summary.entry(key).or_default().push(d);
    }
    println!("  Interleaved 配對表（pair_idx <-> (axis_left, axis_right)):");
    for ((a1, a2), pairs) in &summary {
        if pairs.len() <= 6 {
            println!("    axis ({a1},{a2}): pairs={pairs:?}");
        } else {
            println!(
                "    axis ({a1},{a2}): {} pairs, e.g. {:?} ... {:?}",
                pairs.len(),
                &pairs[..3],
                &pairs[pairs.len() - 3..]
            );
        }
    }
    println!();
    println!("  (完整配對規則已記錄至 docs/rope_pairing_verification.md)");
}
