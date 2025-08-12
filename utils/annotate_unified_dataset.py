import json
from openai import OpenAI
import os
from dotenv import load_dotenv 
from tqdm import tqdm
import time
from typing import List, Dict, Optional


load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("Error: OpenAI API Key not provided.")

client = OpenAI(api_key=api_key,
                base_url="https://api.deepseek.com/v1",)



def build_prompt(entry: Dict) -> str:
    return f"""
You are an expert linguistic and cultural reasoning annotator working on Bengali riddles. Your task is to analyze each riddle-answer pair and assign five metadata labels based on detailed reasoning.

Each riddle is short, figurative, and rooted in Bengali cultural or linguistic logic. Think carefully before assigning labels.

Here is the annotation schema:

1. reasoning_type (Required): What kind of thinking is needed to solve the riddle? Choose exactly one:
   - "metaphorical" – Uses metaphor, analogy, or symbolic comparison
   - "commonsense" – Requires everyday knowledge or practical logic
   - "descriptive" – Physical/visual traits lead directly to answer
   - "wordplay" – Relies on puns, rhyme, or linguistic ambiguity
   - "symbolic" – Depends on cultural symbols, myths, archetypes

2. answer_type (Required): What kind of entity is the answer? Choose exactly one:
   - "object" – Tangible, physical item
   - "person" – Human or character (real or fictional)
   - "nature" – Natural element (e.g., sun, wind)
   - "concept" – Abstract idea (e.g., time, truth)
   - "quantity" – Number or measurable value

3. difficulty (Required): How challenging is the riddle for an average Bengali speaker?
   - "easy" – Very direct; answer is obvious
   - "medium" – Requires moderate abstraction or reasoning
   - "hard" – Requires deep metaphor, cultural or symbolic decoding

4. trap_type (Required): What might mislead a model or person?
   - "surface-literal" – Description tempts overly literal interpretation
   - "ambiguous" – Several plausible interpretations
   - "culturally specific" – The riddle’s solution or logic requires cultural knowledge (e.g., idioms, folk terms, local humor), even if the surface setup seems generic.
   - "none" – No major trap; straightforward

5. source (Optional): Origin of the riddle. Always use "web".

---

Now annotate the following:

Riddle: {entry['question']}
Answer: {entry['answer']}

---

Output **only the following JSON** with no additional text, explanations, or formatting:

{{
  "reasoning_type": "...",
  "answer_type": "...",
  "difficulty": "...",
  "trap_type": "...",
  "source": "web"
}}

Do not include any other text, reasoning, or markdown symbols like json or . Only output the raw JSON.
"""

def annotate_riddle(entry: Dict, max_retries: int = 3) -> Optional[Dict]:
    prompt = build_prompt(entry)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            annotation_str = response.choices[0].message.content.strip()
            annotation = json.loads(annotation_str)
            return {**entry, **annotation}
            
        except json.JSONDecodeError:
            print(f"\n⚠️ JSON decode error on entry ID {entry.get('id')}, attempt {attempt + 1}/{max_retries}")
            time.sleep(1)  # Brief delay before retry
            continue
            
        except Exception as e:
            print(f"\n⚠️ Unexpected error on entry ID {entry.get('id')}: {str(e)}")
            time.sleep(2)  # Longer delay for other errors
            continue
    
    print(f"\n❌ Failed to annotate entry ID {entry.get('id')} after {max_retries} attempts")
    return None

def process_dataset(dataset: List[Dict]) -> List[Dict]:
    annotated_dataset = []
    failed_entries = []
    
    for entry in tqdm(dataset, desc="Annotating riddles", unit="riddle"):
        annotated_entry = annotate_riddle(entry)
        
        if annotated_entry:
            annotated_dataset.append(annotated_entry)
        else:
            failed_entries.append(entry.get("id", "unknown"))
    
    if failed_entries:
        print(f"\n⚠️ Failed to annotate {len(failed_entries)} entries: {failed_entries}")
    
    return annotated_dataset

def main():
    # Load input file
    try:
        with open("deduplicated_dataset_cleaned.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load input file: {str(e)}")
        return
    
    # Process dataset
    annotated_dataset = process_dataset(dataset)
    
    # Save output
    try:
        with open("annotated_dataset.json", "w", encoding="utf-8") as f:
            json.dump(annotated_dataset, f, ensure_ascii=False, indent=2)
        
        success_rate = (len(annotated_dataset) / len(dataset)) * 100
        print(f"\n✅ Annotation complete. Success rate: {success_rate:.1f}%")
        print(f"Saved to annotated_dataset.json ({len(annotated_dataset)} entries)")
        
    except Exception as e:
        print(f"❌ Failed to save output: {str(e)}")

if __name__ == "__main__":
    main()
