# P02-T01 Philox4x32-10 RNG alignment

## 目標
- 將 `src/talker/sampling.rs` 的隨機抽樣改為 qwentts.cpp Random123 相容的 Philox4x32-10。
- 以獨立來源產生的已知向量凍結抽樣行為，並以測試保證隨機字串、位元格式、位元流消耗規則不偏離參照。

## 參照設定
- Philox 常數採用 qwentts.cpp 對應值：
  - `PHILOX_M0 = 0xD2511F53`
  - `PHILOX_M1 = 0xCD9E8D57`
  - `PHILOX_W0 = 0x9E3779B9`
  - `PHILOX_W1 = 0xBB67AE85`
- 鍵值採用 64-bit seed 拆分：
  - `key0 = seed as u32`
  - `key1 = (seed >> 32) as u32`
- 計數器格式：
  - `[ctr_lo, 0, subsequence_lo, subsequence_hi]`
- 均勻數值：
  - `uniform = ((r.x as f32) + 0.5f32) * 2^-32`
- 一次 sample 的消耗：
  - 當 `temperature > 0` 時，消耗 1 個 `subsequence`
  - 當 `temperature <= 0` 時，不消耗 subsequence（greedy 採樣不改變狀態）

## 產生方式
- `tools/generate_philox_vectors.py`
  - 不直接呼叫 Rust 實作
  - 來源：`https://github.com/ServeurpersoCom/qwentts.cpp`
  - commit: `82cd05b9f3a175612dc89fd6943e610fab096ef5`
  - 指令：`python tools/generate_philox_vectors.py`
- 檢查：`python tools/generate_philox_vectors.py --check fixtures/alignment/p02_philox_vectors.json`

## 測試對照
- `tests/philox_rng_test.rs` 使用固定向量做以下檢查：
  - Random123 零 key/零 counter raw words 與 uniform bits。
  - 三個以上非零 seed/counter/counter-start 情境。
  - uniform 32-bit bit pattern 與生成功能一致。
  - 隨機抽樣連續進位（每次一個 subsequence）。
  - greedy 不消耗 subsequence 的狀態行為。
