# Bengali-Fig

Bengali-Fig is a curated benchmark of **traditional and modern Bengali figurative QnA** designed to evaluate large language models (LLMs) on **figurative language understanding, cultural reasoning, and symbolic abstraction**.  
It aims to probe reasoning capabilities in a **low-resource language setting** through tasks that require metaphor interpretation, commonsense reasoning, and culturally grounded inference.

---

## 📌 Project Overview

- **Dataset Size:** ~500+ figurative question–answer pairs after deduplication
- **Format:** Structured CSV/JSON with multiple annotation fields
- **Focus:** Figurative comprehension in Bengali figurative questions
- **Use Case:** Evaluating LLMs in zero-shot and few-shot settings
- **Status:** Data collection, cleaning, and AI-assisted annotation completed; human annotation in progress

---

## 📂 Dataset Structure

Each entry contains:

| Field            | Description |
|------------------|-------------|
| `id`             | Unique identifier for the figurative question |
| `question`       | The figurative question text (Bengali) |
| `answer`         | Gold-standard correct answer |
| `reasoning_type` | Type of reasoning needed (metaphorical, commonsense, descriptive, wordplay, symbolic) |
| `answer_type`    | Type of entity for the answer (object, person, nature, concept, quantity) |
| `difficulty`     | Difficulty level (easy, medium, hard) |
| `trap_type`      | Misdirection type (surface-literal, ambiguous, culturally specific, none) |
| `source`         | figurative question origin (web, book, oral, YouTube, unknown) |
| `answer2`        | Optional alternative correct answer(s) |

---

## 📜 Current Progress

1. **Data Collection:**  
   - Aggregated from web sources.
   - Deduplication performed.

2. **Data Cleaning & Structuring:**  
   - Converted into structured JSON and CSV formats.
   - UTF-8 encoding maintained.

3. **Annotation:**  
   - **Initial AI-assisted annotation** performed using Deepseek with a detailed schema.
   - **Annotation guidelines** prepared for human annotators.
   - **Native speaker annotation** currently in progress for quality assurance.

4. **Next Steps:**  
   - LLM evaluation in **zero-shot** and **few-shot** settings.
   - Model comparison (open-source and proprietary).
   - Error analysis by reasoning type, difficulty, and trap category.

---

## 📊 Planned Evaluation

LLMs will be tested with:
- **Zero-shot prompting**
- **Few-shot prompting**
- **Chain-of-thought reasoning**
- **Comparison across reasoning categories**

Models under consideration:
- GPT
- Claude 
- Gemini
- DeepSeek
- Selected open-source models

---