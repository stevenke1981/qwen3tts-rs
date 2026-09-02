//! # 文本語言自動偵測模組 (Language Detector)
//!
//! 根據輸入文字的 Unicode 碼點分佈、特徵字元與語言詞彙，
//! 高精度判定文字語言（中文、英文、日文、韓文、德文、法文、西文、義文、葡文、俄文等）。

/// 根據文字內容自動偵測最可能的語言標準標籤（如 "chinese", "english", "japanese", "korean", 等）
pub fn detect_language_from_text(text: &str) -> &'static str {
    let text = text.trim();
    if text.is_empty() {
        return "chinese"; // 預設中文
    }

    let mut cjk_count = 0usize;
    let mut hiragana_katakana_count = 0usize;
    let mut hangul_count = 0usize;
    let mut cyrillic_count = 0usize;
    let mut latin_count = 0usize;

    // 特徵字元計數
    let mut german_char_count = 0usize;
    let mut french_char_count = 0usize;
    let mut spanish_char_count = 0usize;
    let mut portuguese_char_count = 0usize;

    for ch in text.chars() {
        match ch {
            // 日文平假名與片假名
            '\u{3040}'..='\u{309F}' | '\u{30A0}'..='\u{30FF}' | '\u{31F0}'..='\u{31FF}' => {
                hiragana_katakana_count += 1;
            }
            // 韓文諺文
            '\u{AC00}'..='\u{D7AF}' | '\u{1100}'..='\u{11FF}' | '\u{3130}'..='\u{318F}' => {
                hangul_count += 1;
            }
            // CJK 統一漢字
            '\u{4E00}'..='\u{9FFF}'
            | '\u{3400}'..='\u{4DBF}'
            | '\u{20000}'..='\u{2A6DF}'
            | '\u{F900}'..='\u{FAFF}' => {
                cjk_count += 1;
            }
            // 西里爾字母（俄文）
            '\u{0400}'..='\u{04FF}' | '\u{0500}'..='\u{052F}' => {
                cyrillic_count += 1;
            }
            // 拉丁字母
            'a'..='z' | 'A'..='Z' => {
                latin_count += 1;
            }
            // 德語特徵字元
            'ä' | 'ö' | 'ü' | 'ß' | 'Ä' | 'Ö' | 'Ü' => {
                latin_count += 1;
                german_char_count += 2;
            }
            // 法語特徵字元
            'é' | 'è' | 'ê' | 'ë' | 'à' | 'â' | 'ù' | 'û' | 'ç' | 'œ' | 'æ' | 'î' | 'ï' | 'ô'
            | 'É' | 'È' | 'Ê' | 'Ë' | 'À' | 'Â' | 'Ù' | 'Û' | 'Ç' | 'Î' | 'Ï' | 'Ô' => {
                latin_count += 1;
                french_char_count += 2;
            }
            // 西班牙語特徵字元
            'ñ' | 'Ñ' | '¿' | '¡' | 'í' | 'ó' | 'ú' | 'Í' | 'Ó' | 'Ú' => {
                latin_count += 1;
                spanish_char_count += 2;
            }
            // 葡萄牙語特徵字元
            'ã' | 'õ' | 'Ã' | 'Õ' => {
                latin_count += 1;
                portuguese_char_count += 2;
            }
            _ => {}
        }
    }

    // 1. 日文判定：只要有出現假名，通常即為日文（因為日文是漢字+假名混合）
    if hiragana_katakana_count > 0 {
        return "japanese";
    }

    // 2. 韓文判定：含有韓文諺文
    if hangul_count > 0 {
        return "korean";
    }

    // 3. 俄文判定：西里爾字母佔主要
    if cyrillic_count > 0 && cyrillic_count >= latin_count {
        return "russian";
    }

    // 4. 中文判定：CJK 漢字數大於等於拉丁字母數，或具有明確漢字
    if cjk_count > 0 && (cjk_count * 2 >= latin_count || latin_count == 0) {
        return "chinese";
    }

    // 5. 若拉丁文字為主，進一步檢查歐洲語系或英語
    if latin_count > 0 {
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower
            .split(|c: char| !c.is_alphabetic())
            .filter(|w| !w.is_empty())
            .collect();

        let mut de_score = german_char_count;
        let mut fr_score = french_char_count;
        let mut es_score = spanish_char_count;
        let mut pt_score = portuguese_char_count;
        let mut it_score = 0usize;

        for &w in &words {
            match w {
                // 德文常用詞
                "der" | "die" | "das" | "und" | "ist" | "ein" | "eine" | "nicht" | "sie"
                | "wir" | "ich" | "mit" | "auf" | "für" | "von" | "aber" => de_score += 2,
                // 法文常用詞
                "le" | "la" | "les" | "des" | "un" | "une" | "est" | "et" | "que" | "qui"
                | "pour" | "dans" | "avec" | "vous" | "nous" | "sur" => fr_score += 2,
                // 西文常用詞
                "el" | "los" | "las" | "del" | "por" | "para" | "con" | "como" | "pero"
                | "este" | "esta" | "son" => es_score += 2,
                // 義文常用詞
                "il" | "lo" | "gli" | "nel" | "nella" | "della" | "sono" | "questo" | "perché"
                | "tutto" | "anche" | "cosa" => it_score += 2,
                // 葡文常用詞
                "não" | "são" | "uma" | "mais" | "muito" | "pelo" | "pela" | "com" => {
                    pt_score += 2
                }
                _ => {}
            }
        }

        let max_score = de_score
            .max(fr_score)
            .max(es_score)
            .max(it_score)
            .max(pt_score);
        if max_score >= 2 {
            if max_score == de_score {
                return "german";
            }
            if max_score == fr_score {
                return "french";
            }
            if max_score == es_score {
                return "spanish";
            }
            if max_score == it_score {
                return "italian";
            }
            if max_score == pt_score {
                return "portuguese";
            }
        }

        return "english";
    }

    // 預設中文
    "chinese"
}

/// 將標準語言代碼或名稱轉換為易讀的 UI 顯示名稱
pub fn language_display_name(lang: &str) -> &'static str {
    match lang.trim().to_lowercase().as_str() {
        "auto" => "自動偵測",
        "zh" | "chinese" | "zh-cn" | "zh-tw" | "mandarin" => "中文 (Chinese)",
        "en" | "english" => "英語 (English)",
        "ja" | "japanese" => "日語 (Japanese)",
        "ko" | "korean" => "韓語 (Korean)",
        "de" | "german" => "德語 (German)",
        "fr" | "french" => "法語 (French)",
        "es" | "spanish" => "西班牙語 (Spanish)",
        "it" | "italian" => "義大利語 (Italian)",
        "pt" | "portuguese" => "葡萄牙語 (Portuguese)",
        "ru" | "russian" => "俄語 (Russian)",
        "beijing_dialect" => "北京話 (Beijing Dialect)",
        "sichuan_dialect" => "四川話 (Sichuan Dialect)",
        _ => "其他語言",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_chinese() {
        assert_eq!(detect_language_from_text("你好，世界！今天天氣真好。"), "chinese");
        assert_eq!(detect_language_from_text("Qwen3-TTS 語音合成系統"), "chinese");
    }

    #[test]
    fn test_detect_japanese() {
        assert_eq!(detect_language_from_text("こんにちは、世界！"), "japanese");
        assert_eq!(detect_language_from_text("これは日本語のテストです。"), "japanese");
        assert_eq!(detect_language_from_text("ラーメンが食べたい"), "japanese");
    }

    #[test]
    fn test_detect_korean() {
        assert_eq!(detect_language_from_text("안녕하세요, 세계!"), "korean");
        assert_eq!(detect_language_from_text("한국어 음성 합성 테스트"), "korean");
    }

    #[test]
    fn test_detect_english() {
        assert_eq!(detect_language_from_text("Hello, world! How are you today?"), "english");
        assert_eq!(
            detect_language_from_text("This is an end-to-end text-to-speech synthesis."),
            "english"
        );
    }

    #[test]
    fn test_detect_german() {
        assert_eq!(
            detect_language_from_text("Guten Tag! Das ist ein schöner Tag für uns alle."),
            "german"
        );
        assert_eq!(detect_language_from_text("Ich möchte ein großes Bier bitte."), "german");
    }

    #[test]
    fn test_detect_french() {
        assert_eq!(
            detect_language_from_text("Bonjour le monde! C'est un plaisir de vous rencontrer."),
            "french"
        );
        assert_eq!(detect_language_from_text("La vie est belle avec la musique."), "french");
    }

    #[test]
    fn test_detect_spanish() {
        assert_eq!(
            detect_language_from_text("¡Hola! ¿Cómo estás hoy? El tiempo está muy bueno."),
            "spanish"
        );
    }

    #[test]
    fn test_detect_russian() {
        assert_eq!(
            detect_language_from_text("Привет, мир! Это тест синтеза речи."),
            "russian"
        );
    }

    #[test]
    fn test_detect_empty_or_whitespace() {
        assert_eq!(detect_language_from_text(""), "chinese");
        assert_eq!(detect_language_from_text("   \n\t  "), "chinese");
    }
}
