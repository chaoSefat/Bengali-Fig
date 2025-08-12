# 🧠 BengaliFig Human Annotation Guidelines
## 🔍 Objective

Evaluate the correctness and consistency of LLM-assigned metadata for Bengali riddles. Each riddle is annotated with 5 fields. Your task is to verify or correct them based on the definitions below.

### 1. reasoning_type (Required)

Definition: What kind of cognitive process is required to solve the riddle?

metaphorical – Requires interpreting metaphor, analogy, or symbolic comparison.
e.g., “black umbrella in the sky” → cloud

commonsense – Requires real-world or everyday functional knowledge.
e.g., “has no legs but runs all day” → river

descriptive – Solution is derived directly from physical traits described.
e.g., “round, red, juicy” → apple

wordplay – Involves puns, phonetic tricks, rhymes, or linguistic ambiguity.
e.g., Bengali double meanings

symbolic – Relies on culturally encoded symbols or archetypes.
e.g., “Ma Durga’s vehicle” → lion

✅ Guidance: Choose the type that best explains why the riddle is difficult to answer literally. Prioritize metaphorical/symbolic over commonsense if both apply.

### 2. answer_type (Required)

Definition: What kind of entity is the answer?

object – Physical, tangible thing (e.g., clock, shoe, knife)

person – Human being, role, or fictional character (e.g., father, thief)

nature – Natural entity (e.g., river, moon, wind)

concept – Abstract idea (e.g., time, truth, love)

quantity – Numerical or countable value (e.g., one, hundred, none)

✅ Guidance: Classify the answer itself, not the description.

### 3. difficulty (Required)

Definition: How challenging is the riddle for an average adult Bengali speaker?

easy – Direct clues, little abstraction

medium – Some abstraction or reasoning required

hard – Requires deep metaphor, cultural knowledge, or clever inference

✅ Guidance: Use your intuition but err on the conservative side. Think about whether the riddle would confuse a non-expert.

### 4. trap_type (Required)

Definition: What kind of misdirection is built into the riddle?

surface-literal – Descriptions lead to an overly literal answer

ambiguous – Multiple valid interpretations possible

culturally specific – Requires cultural or local knowledge to resolve

none – No real trap; straightforward or descriptive

✅ Guidance: Focus on what might mislead a language model or reader.

### 5. source (Optional)

Choose from: web, book, oral, YouTube, or unknown

✅ Guidance: Use the provided metadata or guess if clear (e.g., ripped from a video = YouTube).

✅ Review Protocol
Read the riddle and answer.

Check each field one by one.

If correct, leave unchanged. If wrong, revise it.

Flag any riddle that is unclear, nonsensical, or mistranslated.

### Common Error Fixes

Error Fix
Metaphorical vs. symbolic confused Use symbolic if culturally grounded (e.g., deity symbols)
Object vs. concept mix-up “Clock” = object, “Time” = concept
Overestimated difficulty Many metaphorical riddles are still medium, not hard
Trap not labeled Most riddles have at least a surface-literal or cultural trap

### Output Format

Use a spreadsheet or form with these columns:

question

answer

reasoning_type

answer_type

difficulty

trap_type

source

needs_review (✔️ if unsure)
