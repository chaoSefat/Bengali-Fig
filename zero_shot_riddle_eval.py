#!/usr/bin/env python3
"""
Bengali Riddle LLM Evaluation Script
Zero-shot evaluation of multiple LLMs on Bengali riddles dataset
"""

import json
import os
import argparse
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path
import re

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RiddleEvaluator:
    def __init__(self, provider: str, model: str, data_file: str, output_dir: str = "results"):
        """
        Initialize the evaluator
        
        Args:
            provider: API provider ('openai', 'deepseek', 'anthropic')
            model: Model identifier
            data_file: Path to JSON file containing riddles
            output_dir: Directory to save results
        """
        self.provider = provider.lower()
        self.model = model
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Initialize client
        self.client = self._initialize_client()
        
        # Results storage
        self.results = []
        self.processed_count = 0
        
        # Create unique filename for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.output_dir / f"{self.provider}_{self.model.replace('/', '_')}_{timestamp}.json"
        
    def _initialize_client(self) -> OpenAI:
        """Initialize the appropriate API client"""
        if self.provider == 'openai':
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Error: OPENAI_API_KEY not found in environment variables")
            return OpenAI(api_key=api_key)
            
        elif self.provider == 'novita':
            api_key = os.getenv("NOVITA_API_KEY")
            if not api_key:
                raise ValueError("Error: NOVITA_API_KEY not found in environment variables")
            return OpenAI(
                api_key=api_key,
                base_url="https://api.novita.ai/openai"
            )
        
        elif self.provider == 'deepseek':
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("Error: DEEPSEEK_API_KEY not found in environment variables")
            return OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )
            
        elif self.provider == 'anthropic':
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Error: ANTHROPIC_API_KEY not found in environment variables")
            return OpenAI(
                api_key=api_key,
                base_url="https://api.anthropic.com/v1/"
            )
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate_zero_shot_prompt(self, riddle_data: Dict[str, Any]) -> str:
        """Generate zero-shot prompt for a riddle"""
        question = riddle_data['question']
        options = riddle_data['options']
        
        # Format options as A, B, C, D
        option_text = ""
        for i, option in enumerate(options):
            letter = chr(65 + i)  # A, B, C, D
            option_text += f"{letter}) {option}\n"
        
        prompt = f"""নিচের ধাঁধাটি সমাধান করুন এবং সঠিক উত্তরের এক অক্ষরে (A, B, C, অথবা D) দিন:

প্রশ্ন: {question}

বিকল্পসমূহ:
{option_text.strip()} 

শুধু JSON আকারে উত্তর দিন. কোনো ব্যাখ্যা বা  বর্ণনা দেবেন না. উদাহরণস্বরূপ:
{{"উত্তর": "<আপনার উত্তর এখানে>"}}
"""
        
        return prompt
    
    def call_model(self, prompt: str) -> str:
        """Make API call to the model"""
        try:
            # Special handling for GPT-5 (no temperature parameter)
            if self.model.lower() == 'gpt-5':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        # {"role": "system", "content": "আপনি একটি উত্তরকারী বট। সর্বদা শুধুমাত্র একটি অক্ষর (A, B, C, D) দিয়ে উত্তর দিন। কোনো ব্যাখ্যা লিখবেন না।"},
                        {"role": "user", "content": prompt}
                              ]
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        # {"role": "system", "content": "আপনি একটি উত্তরকারী বট। সর্বদা শুধুমাত্র একটি অক্ষর (A, B, C, D) দিয়ে উত্তর দিন। কোনো ব্যাখ্যা লিখবেন না।"},
                        {"role": "user", "content": prompt}],
                    temperature=0
                )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error calling model: {e}")
            return "ERROR"
    
    def extract_answer(self, response: str) -> Tuple[str, str]:
        """
        Extract the predicted answer from model response by first attempting JSON parsing,
        then falling back to regex.
        
        Returns:
            Tuple of (predicted_letter, confidence_level)
        """
        response = response.strip()
        
        # 1. Attempt to parse as JSON
        try:
            data = json.loads(response)
            # Check for both 'answer' and 'উত্তর' keys
            predicted_option = data.get('answer', data.get('উত্তর', '')).strip().upper()
            if re.match(r'^[ABCD]$', predicted_option):
                return predicted_option, "high"
        except (json.JSONDecodeError, AttributeError):
            # JSON parsing failed, fall back to regex
            pass
            
        # 2. Fallback to regex-based extraction
        
        # Look for single letter answers (A, B, C, D)
        single_letter_match = re.search(r'^([ABCD])$', response.upper())
        if single_letter_match:
            return single_letter_match.group(1), "high"
        
        # Look for letter followed by closing parenthesis
        paren_match = re.search(r'^([ABCD])\)', response.upper())
        if paren_match:
            return paren_match.group(1), "high"
        
        # Look for "উত্তর: X" or "Answer: X" pattern
        answer_pattern = re.search(r'(?:উত্তর|Answer)[:\s]*([ABCD])', response, re.IGNORECASE)
        if answer_pattern:
            return answer_pattern.group(1).upper(), "medium"
        
        # Look for any A, B, C, or D in the response
        letter_match = re.search(r'([ABCD])', response.upper())
        if letter_match:
            return letter_match.group(1), "low"
        
        # If no clear answer found
        return "UNKNOWN", "none"
    
    
    def process_riddle(self, delay, riddle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single riddle and return result"""
        riddle_id = riddle_data['id']
        
        # Generate prompt
        prompt = self.generate_zero_shot_prompt(riddle_data)

        # wait a bit before calling the model to avoid rate limits
        time.sleep(5)
        
        # Get model response
        raw_response = self.call_model(prompt)
        
        # Extract prediction
        predicted_letter, confidence = self.extract_answer(raw_response)
        
        # Determine correctness
        correct_letter = riddle_data['correct_option']
        is_correct = predicted_letter == correct_letter
        
        # Compile result
        result = {
            'riddle_id': riddle_id,
            'question': riddle_data['question'],
            'correct_answer': riddle_data['answer'],
            'correct_option': correct_letter,
            'predicted_option': predicted_letter,
            'is_correct': is_correct,
            'confidence': confidence,
            'raw_response': raw_response,
            'reasoning_type': riddle_data['reasoning_type'],
            'answer_type': riddle_data['answer_type'],
            'difficulty': riddle_data['difficulty'],
            'trap_type': riddle_data['trap_type'],
            'cultural_depth': riddle_data['cultural_depth'],
            'options': riddle_data['options'],
            'distractors': riddle_data['distractors'],
            'provider': self.provider,
            'model': self.model,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def save_results(self, append_mode: bool = False):
        """Save results to file"""
        # Create metadata
        metadata = {
            'provider': self.provider,
            'model': self.model,
            'total_riddles': len(self.data),
            'processed_count': self.processed_count,
            'accuracy': self.calculate_accuracy(),
            'timestamp': datetime.now().isoformat(),
            'data_file': str(self.data_file)
        }
        
        # Combine metadata and results
        output_data = {
            'metadata': metadata,
            'results': self.results
        }
        
        # Save to file
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {self.results_file}")
    
    def calculate_accuracy(self) -> Dict[str, float]:
        """Calculate various accuracy metrics"""
        if not self.results:
            return {}
        
        total = len(self.results)
        correct = sum(1 for r in self.results if r['is_correct'])
        
        # Overall accuracy
        overall_accuracy = correct / total if total > 0 else 0
        
        # Accuracy by reasoning type
        reasoning_accuracy = {}
        for reasoning_type in set(r['reasoning_type'] for r in self.results):
            reasoning_results = [r for r in self.results if r['reasoning_type'] == reasoning_type]
            reasoning_correct = sum(1 for r in reasoning_results if r['is_correct'])
            reasoning_accuracy[reasoning_type] = reasoning_correct / len(reasoning_results)
        
        # Accuracy by difficulty
        difficulty_accuracy = {}
        for difficulty in set(r['difficulty'] for r in self.results):
            difficulty_results = [r for r in self.results if r['difficulty'] == difficulty]
            difficulty_correct = sum(1 for r in difficulty_results if r['is_correct'])
            difficulty_accuracy[difficulty] = difficulty_correct / len(difficulty_results)
        
        return {
            'overall': overall_accuracy,
            'by_reasoning_type': reasoning_accuracy,
            'by_difficulty': difficulty_accuracy,
            'total_processed': total,
            'total_correct': correct
        }
    
    def run_evaluation(self,delay=3,batch_delay=5, start_index = 0):
        """Run the complete evaluation"""
        print(f"Starting evaluation...")
        print(f"Provider: {self.provider}")
        print(f"Model: {self.model}")
        print(f"Total riddles: {len(self.data)}")
        print(f"Results will be saved to: {self.results_file}")
        print(f"Starting from index: {start_index}")
        print(f"Delay between requests: {delay} seconds")
        print(f"Batch delay every 10 riddles: {batch_delay} seconds")
        print("-" * 50)
        
        for i, riddle in enumerate(self.data):
            if i < start_index:
                continue  # Skip already processed riddles

            print(f"Processing riddle {i+1}/{len(self.data)} (ID: {riddle['id']})")
            
            # Process riddle
            result = self.process_riddle(delay,riddle)
            self.results.append(result)
            self.processed_count += 1
            
            # Print immediate result
            status = "✓" if result['is_correct'] else "✗"
            print(f"  {status} Predicted: {result['predicted_option']}, Correct: {result['correct_option']}")
            
            # Save every 10 riddles
            if (i + 1) % 10 == 0:
                self.save_results()
                accuracy = self.calculate_accuracy()
                print(f"  Checkpoint: {accuracy['total_correct']}/{accuracy['total_processed']} correct ({accuracy['overall']:.2%})")
                print()
            
            # Small delay to be respectful to APIs
            time.sleep(10)
        
        # Final save
        self.save_results()
        
        # Print final statistics
        print("\n" + "=" * 50)
        print("EVALUATION COMPLETE!")
        print("=" * 50)
        
        final_accuracy = self.calculate_accuracy()
        print(f"Overall Accuracy: {final_accuracy['overall']:.2%} ({final_accuracy['total_correct']}/{final_accuracy['total_processed']})")
        
        print("\nAccuracy by Reasoning Type:")
        for reasoning_type, acc in final_accuracy['by_reasoning_type'].items():
            print(f"  {reasoning_type}: {acc:.2%}")
        
        print("\nAccuracy by Difficulty:")
        for difficulty, acc in final_accuracy['by_difficulty'].items():
            print(f"  {difficulty}: {acc:.2%}")
        
        print(f"\nResults saved to: {self.results_file}")

def main():
   

    settings = [

            {
                'provider': 'novita',
                'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
                'delay': 5,
                'batch_delay': 10
            }
            # ,
            # {
            #     'provider': 'novita',
            #     'model': 'meta-llama/llama-4-maverick-17b-128e-instruct-fp8',
            #     'delay': 5,
            #     'batch_delay': 10
            # }
            # 
            # ,

            # {
            #     'provider': 'novita',
            #     'model': 'qwen/qwen3-235b-a22b-instruct-2507',
            #     'delay': 5,
            #     'batch_delay': 10
            # }
            # ,
            ## Done
            # {
            #     'provider': 'deepseek',
            #     'model': 'deepseek-chat',
            #     'delay': 5,
            #     'batch_delay': 10
            # },
            ## Done 
            # {
            #     'provider': 'openai',
            #     'model': 'gpt-4.1',
            #     'delay': 5,
            #     'batch_delay': 10
            # }
            # ,
            # {
            #     'provider': 'openai',
            #     'model': 'gpt-5',
            #     'delay': 5,
            #     'batch_delay': 10
            # }
            # ,
            # {
            #     'provider': 'anthropic',
            #     'model': 'claude-sonnet-4-0',
            #     'delay': 5,
            #     'batch_delay': 10
            # },
            # {
            #     'provider': 'anthropic',
            #     'model': 'claude-opus-4-1',
            #     'delay': 5,
            #     'batch_delay': 10
            # }
            # ,
            # {
            #     'provider': 'deepseek',
            #     'model': 'deepseek-reasoner',
            #     'delay': 5,
            #     'batch_delay': 10
            # }
        ]
    


    data_file = "data/v4_patched_mcq_dataset.json"
    output_dir = "results/zero_shot"

    # data_file = "sandbox/sandbox.json"
    # output_dir = "sandbox/results"

    for setting in settings:
        provider = setting['provider']
        model = setting['model']
        evaluator = RiddleEvaluator(
            provider=provider,
            model=model,
            data_file=data_file,
            output_dir=output_dir
        )
        start_index = 0
        evaluator.run_evaluation(setting['delay'],setting['batch_delay'],start_index=start_index)

if __name__ == "__main__":
    main()