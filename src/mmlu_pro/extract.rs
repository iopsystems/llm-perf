use regex::Regex;

/// Strip thinking/reasoning tags from model output before extraction.
/// Handles <think>...</think>, <reasoning>...</reasoning>, etc.
fn strip_thinking_tags(text: &str) -> String {
    let tags = ["think", "thinking", "reasoning", "reflection"];
    let mut result = text.to_string();
    for tag in tags {
        let pattern = format!(r"(?si)<{}>.*?</{}>", tag, tag);
        let re = Regex::new(&pattern).unwrap();
        result = re.replace_all(&result, "").to_string();
    }
    result
}

/// Three-stage answer extraction matching the original TIGER-Lab logic,
/// with thinking-model support (strip tags first).
///
/// Returns None on failure — no random guessing.
pub fn extract_answer(text: &str) -> Option<char> {
    let cleaned = strip_thinking_tags(text);
    let text = cleaned.trim();

    // Stage 1: "answer is (X)" pattern
    if let Some(answer) = extract_answer_is(text) {
        return Some(answer);
    }

    // Stage 2: "Answer: X" pattern
    if let Some(answer) = extract_answer_colon(text) {
        return Some(answer);
    }

    // Stage 3: Last standalone A-J
    extract_last_letter(text)
}

/// Stage 1: Match "answer is (X)" or "answer is X"
fn extract_answer_is(text: &str) -> Option<char> {
    let re = Regex::new(r"(?i)answer is \(?([A-J])\)?").unwrap();
    re.captures(text)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().chars().next())
}

/// Stage 2: Match "Answer: X", "answer: X", "ANSWER: X", etc.
fn extract_answer_colon(text: &str) -> Option<char> {
    let re = Regex::new(r"(?i)answer:\s*([A-J])").unwrap();
    re.captures(text)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().chars().next())
}

/// Stage 3: Last standalone capital letter A-J in the text
fn extract_last_letter(text: &str) -> Option<char> {
    let re = Regex::new(r"\b([A-J])\b").unwrap();
    let mut last = None;
    for caps in re.captures_iter(text) {
        if let Some(m) = caps.get(1) {
            last = m.as_str().chars().next();
        }
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_answer_is() {
        assert_eq!(extract_answer("the answer is (A)"), Some('A'));
        assert_eq!(extract_answer("the answer is B"), Some('B'));
        assert_eq!(extract_answer("The Answer is (C)"), Some('C'));
    }

    #[test]
    fn test_extract_answer_colon() {
        assert_eq!(extract_answer("Answer: D"), Some('D'));
        assert_eq!(extract_answer("answer: E"), Some('E'));
        assert_eq!(extract_answer("Answer:  F"), Some('F'));
        assert_eq!(extract_answer("ANSWER: G"), Some('G'));
        assert_eq!(extract_answer("ANSWER:  H"), Some('H'));
    }

    #[test]
    fn test_extract_last_letter() {
        assert_eq!(extract_answer("I think it's B or maybe A"), Some('A'));
        assert_eq!(extract_answer("Looking at option G"), Some('G'));
    }

    #[test]
    fn test_extract_none() {
        assert_eq!(extract_answer(""), None);
        assert_eq!(extract_answer("no capital letters here at all"), None);
        // "I" is a standalone A-J letter, so stage 3 picks it up
        assert_eq!(extract_answer("I don't know the answer"), Some('I'));
    }

    #[test]
    fn test_strip_thinking_tags() {
        let text = "<think>Let me reason about this... the answer might be A but actually B</think>The answer is (C)";
        assert_eq!(extract_answer(text), Some('C'));
    }

    #[test]
    fn test_strip_reasoning_tags() {
        let text = "<reasoning>Complex analysis pointing to B</reasoning>\nAnswer: D";
        assert_eq!(extract_answer(text), Some('D'));
    }

    #[test]
    fn test_strip_multiline_thinking() {
        let text = "<think>\nLine 1\nLine 2\nthe answer is A\n</think>\nthe answer is (B)";
        assert_eq!(extract_answer(text), Some('B'));
    }
}
