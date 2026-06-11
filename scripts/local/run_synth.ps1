$env:QWEN3_TTS_MODEL_DIR = 'C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-Base\snapshots\fd4b254389122332181a7c3db7f27e918eec64e3'
$env:QWEN3_TTS_TOKENIZER = 'D:\qwen3tts-rs\models\tokenizer_1.7b.json'
Set-Location D:\qwen3tts-rs
cargo run --example synthesize --features candle-llm --release --quiet -- --text "今天天氣真好" --backend candle --language chinese --output cn_candle_1.7b_short.wav --max-new-tokens 96 *>&1 | Tee-Object -FilePath last_run.log
