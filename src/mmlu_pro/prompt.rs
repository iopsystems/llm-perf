use llm_perf::client::Message;

use super::dataset::Question;

const CHOICE_MAP: &[char] = &['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'];

/// Format a question and its options into the expected text format.
/// Returns (formatted_question, cot_content).
fn format_example(question: &str, options: &[String], cot_content: &str) -> (String, String) {
    let cot = if cot_content.is_empty() {
        "Let's think step by step.".to_string()
    } else if let Some(stripped) = cot_content.strip_prefix("A: ") {
        stripped.trim().to_string()
    } else {
        cot_content.trim().to_string()
    };

    let mut example = format!("Question: {}\nOptions: ", question);
    for (i, opt) in options.iter().enumerate() {
        if i < CHOICE_MAP.len() {
            example.push_str(&format!("{}. {}\n", CHOICE_MAP[i], opt));
        }
    }

    (example.trim().to_string(), cot)
}

/// Build multi-chat prompt messages for a question.
///
/// Structure:
/// - System message with {subject} substituted
/// - For each CoT example: user message (question + options) → assistant message (Answer: cot)
/// - Final user message with the test question
pub fn build_messages(
    system_prompt: &str,
    cot_examples: &[Question],
    question: &str,
    options: &[String],
) -> Vec<Message> {
    let mut messages = Vec::new();

    // System message
    messages.push(Message {
        role: "system".to_string(),
        content: system_prompt.to_string(),
    });

    // CoT examples as multi-turn conversation
    for example in cot_examples {
        let (formatted, cot_content) =
            format_example(&example.question, &example.options, &example.cot_content);
        messages.push(Message {
            role: "user".to_string(),
            content: formatted,
        });
        messages.push(Message {
            role: "assistant".to_string(),
            content: format!("Answer: {}", cot_content),
        });
    }

    // Test question
    let (formatted, _) = format_example(question, options, "");
    messages.push(Message {
        role: "user".to_string(),
        content: formatted,
    });

    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_example_basic() {
        let options = vec!["Yes".to_string(), "No".to_string()];
        let (example, cot) =
            format_example("Is the sky blue?", &options, "Let's think step by step.");
        assert!(example.contains("Question: Is the sky blue?"));
        assert!(example.contains("A. Yes"));
        assert!(example.contains("B. No"));
        assert_eq!(cot, "Let's think step by step.");
    }

    #[test]
    fn test_format_example_strips_a_prefix() {
        let options = vec!["opt1".to_string()];
        let (_, cot) = format_example("Q?", &options, "A: The answer is...");
        assert_eq!(cot, "The answer is...");
    }

    #[test]
    fn test_build_messages_structure() {
        let examples = vec![Question {
            question_id: 1,
            question: "Example Q?".to_string(),
            options: vec!["A".to_string(), "B".to_string()],
            answer: "A".to_string(),
            answer_index: 0,
            cot_content: "Let's think...".to_string(),
            category: "math".to_string(),
        }];

        let messages = build_messages(
            "System prompt about {subject}",
            &examples,
            "Test Q?",
            &["X".to_string(), "Y".to_string()],
        );

        assert_eq!(messages.len(), 4); // system + user + assistant + user
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[2].role, "assistant");
        assert!(messages[2].content.starts_with("Answer: "));
        assert_eq!(messages[3].role, "user");
    }
}
