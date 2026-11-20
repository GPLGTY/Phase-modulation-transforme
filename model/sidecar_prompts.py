"""Sidecar generation templates (plain / augmented / letter-only)."""

from __future__ import annotations

SYSTEM = (
    "You are an expert at answering medical multiple-choice questions. "
    "Based on the provided information, select which of the options A through E "
    "best answers the given question."
)

PROMPT_LETTER = """You are answering a multiple-choice medical question.
Output exactly one character: A, B, C, D, or E.
Do not write explanations, punctuation, or any other text.

Question:
{question_with_options}
"""

LEAD = "In response to your question, my analysis process and answer are as follows:"

PROMPT_PLAIN = """You write the user message for a medical multiple-choice example. Output only the user message, in English. No JSON, no system line, no assistant line, no title.

You are NOT given the correct answer. Analyze the question and options using your own medical knowledge, then choose the single best option yourself.

Match this layout exactly:

In response to your question, my analysis process and answer are as follows:

**Analysis:**

<3 to 5 numbered points about the question's key facts. Bold important terms. Do not paste the whole question.>

Then one bullet per option, covering every option exactly once:
- **A. <option text>** – <why it fits or fails>
- **B. <option text>** – ...
- **C. <option text>** – ...
- **D. <option text>** – ...
- **E. <option text>** – ... (include E only if the question has option E)

Copy each option letter and option text exactly from the question. Do not invent, rename, merge, or add options. If the question has no option E, do not write an E bullet. Do not repeat the same sentence.

**Conclusion:**
<2 to 4 sentences. Name the chosen option in a full sentence, including its letter and its text, and the main reason.>

Therefore, the answer to the above question is: 

Length: about 2000 to 5000 characters.
The last line must be exactly "Therefore, the answer to the above question is: " with a space after the colon. Do not put a letter, option text, bold marks, or any other characters on that line. The letter belongs only in the conclusion sentences above it.

Question:
{question_with_options}
"""

PROMPT_AUGMENTED = """You write the user message for a medical multiple-choice example. Output only the user message, in English. No JSON, no system line, no assistant line, no title.

You are NOT given the correct answer. Below are medical knowledge passages retrieved from reference textbooks; they may or may not be relevant — use them only when helpful. Analyze the question and options, then choose the single best option yourself.

Match this layout exactly:

In response to your question, my analysis process and answer are as follows:

**Analysis:**

<3 to 5 numbered points about the question's key facts. Bold important terms. Do not paste the whole question. Do not paste or quote long retrieved passages; paraphrase only what you use.>

Then one bullet per option, covering every option exactly once:
- **A. <option text>** – <why it fits or fails>
- **B. <option text>** – ...
- **C. <option text>** – ...
- **D. <option text>** – ...
- **E. <option text>** – ... (include E only if the question has option E)

Copy each option letter and option text exactly from the question. Do not invent, rename, merge, or add options. If the question has no option E, do not write an E bullet. Do not repeat the same sentence.

**Conclusion:**
<2 to 4 sentences. Name the chosen option in a full sentence, including its letter and its text, and the main reason.>

Therefore, the answer to the above question is: 

Length: about 2000 to 5000 characters.
The last line must be exactly "Therefore, the answer to the above question is: " with a space after the colon. Do not put a letter, option text, bold marks, or any other characters on that line. The letter belongs only in the conclusion sentences above it.
Do NOT include a "#Medical knowledge#" section or any raw retrieved text in your output.

Retrieved knowledge (for your reasoning only; do not copy into the output):
<knowledge>
{knowledges_for_llm}
</knowledge>

Question:
{question_with_options}
"""
