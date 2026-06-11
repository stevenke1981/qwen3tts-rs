$env:RUST_BACKTRACE = '1'
Set-Location -LiteralPath 'D:\qwen3tts-rs'
& 'D:\qwen3tts-rs\target\release\examples\synthesize.exe' `
  --text '今天天氣真好' `
  --language chinese `
  --backend candle `
  --model-dir 'C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-Base\snapshots\fd4b254389122332181a7c3db7f27e918eec64e3' `
  --output 'D:\qwen3tts-rs\cn_candle_1.7b_short.wav' `
  --max-new-tokens 32
