$env:RUST_BACKTRACE = 'full'
Set-Location -LiteralPath 'D:\qwen3tts-rs'
& 'D:\qwen3tts-rs\target\debug\examples\synthesize.exe' `
  --text 'hello' `
  --language english `
  --backend candle `
  --model-dir 'C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-Base\snapshots\fd4b254389122332181a7c3db7f27e918eec64e3' `
  --output 'D:\qwen3tts-rs\cn_candle_1.7b_hello.wav' `
  --max-new-tokens 16
