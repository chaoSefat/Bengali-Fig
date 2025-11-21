#!/usr/bin/env python3
"""
Bengali Riddle CoT (Chain of Thought) Evaluation Script
Evaluates LLMs using chain of thought prompting on Bengali riddles dataset
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

class CoTRiddleEvaluator:
    def __init__(self, provider: str, model: str, data_file: str, output_dir: str = "results"):
        """
        Initialize the CoT evaluator
        
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
        self.results_file = self.output_dir / f"cot_{self.provider}_{self.model.replace('/', '_')}_{timestamp}.json"
        
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
    
    def generate_cot_prompt(self, riddle_data: Dict[str, Any]) -> str:
        """Generate Chain of Thought prompt for a riddle"""
        question = riddle_data['question']
        options = riddle_data['options']
        
        # Format options as A, B, C, D
        option_text = ""
        for i, option in enumerate(options):
            letter = chr(65 + i)  # A, B, C, D
            option_text += f"{letter}) {option}\n"
        
        prompt = f"""
        আপনি একটি বাংলা ধাঁধার বিশেষজ্ঞ। নিম্ন কিছু ধাঁধার উদাহরণ দেওয়া হলো, যেখানে ধাঁধার সমাধান বিশ্লেষণ করা হয়েছে। 
        
        উদাহরণ ১:

        প্রশ্ন: একটা ঘড়ির উপর দিয়ে একটা ঘোড়া চলে গেল, ঘড়িটার কটা বাজবে।
        বিকল্পসমূহ:
        A.সাতটা
        B.বারোটা
        C.নটা
        D.তিনটা

        যুক্তি: ঘড়ির উপর দিয়ে ঘোড়া চললে ঘড়ির কাটা ভেঙে যাবে, আর কাজ করবে না। বারোটা বাজা অর্থ অকেজ বা ধ্বংস হয়ে যাওয়া। তাই ঘড়িটার বারোটা বাজবে।
        উত্তর: B
        
        উদাহরণ ২:
        প্রশ্ন: কোন কার চলে না?
        বিকল্পসমূহ:
        A.নৌকা
        B.সাইকেল
        C.কুকার
        D.গাড়ি
        
        যুক্তি: নৌকা, সাইকেল, গাড়ি এগুলো সবই চলতে পারে এবং যানবাহন তথা "কার", কিন্তু কুকার (রান্নার পাত্র) চলতে পারে না। তাই সঠিক উত্তর হবে কুকার।
        উত্তর: C

        উদাহরণ ৩:
        প্রশ্ন: নাকের ডগায় পৈতে আটকান চৈতনে মার টান গলায় ধরে দাও পটকান ঘুরতে থাকে ঘ্যানের ঘ্যান।
        বিকল্পসমূহ:
        A.হামানদিস্তা
        B.লাট্টু
        C.হাতুড়ি
        D.দা

        উত্তর: B
        যুক্তি: নাকের ডগায় পৈতে (দড়ি) আটকানো এবং টান দিলে গলায় ধরে ঘুরতে থাকা ও শব্দ করা—এই সবকিছুই লাট্টুর বৈশিষ্ট্যের সাথে মিলে যায়। তাই সঠিক উত্তর হবে লাট্টু।
        
        এখন নিচের ধাঁধাটি সমাধান করুন। প্রথমে আপনার যুক্তি ব্যাখ্যা করুন, তারপর উত্তর দিন:

প্রশ্ন: {question}

বিকল্পসমূহ:
{option_text.strip()}

নিম্নলিখিত ধাপগুলো অনুসরণ করুন:

১. **প্রশ্ন বিশ্লেষণ**: প্রশ্নটিতে কী জিজ্ঞাসা করা হচ্ছে তা স্পষ্টভাবে বুঝুন। কোন ধরনের ধাঁধা এটি (যেমন: শব্দের খেলা, যুক্তি, সাংস্কৃতিক জ্ঞান)?

২. **প্রতিটি বিকল্প মূল্যায়ন**: প্রতিটি বিকল্প (A, B, C, D) পর্যালোচনা করুন এবং দেখুন কোনটি সবচেয়ে যুক্তিসঙ্গত মনে হচ্ছে।

৩. **যুক্তিসঙ্গত সিদ্ধান্ত**: আপনার বিশ্লেষণের ভিত্তিতে সবচেয়ে যুক্তিসঙ্গত উত্তর বেছে নিন।

JSON আকারে উত্তর দিন যাতে যুক্তি এবং উত্তর দুটোই থাকে:
{{"যুক্তি": "<আপনার যুক্তি এখানে>", "উত্তর": "<A/B/C/D>"}}

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
                        {"role": "system", "content": "আপনি একটি বাংলা ধাঁধা বিশেষজ্ঞ। সর্বদা ধাপে ধাপে চিন্তা করুন এবং JSON ফরম্যাটে উত্তর দিন।"},
                        {"role": "user", "content": prompt}
                    ]
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "আপনি একটি বাংলা ধাঁধা বিশেষজ্ঞ। সর্বদা ধাপে ধাপে চিন্তা করুন এবং JSON ফরম্যাটে উত্তর দিন।"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error calling model: {e}")
            return "ERROR"
    
    def extract_answer_from_cot(self, response: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Extract the predicted answer and reasoning from CoT response
        
        Returns:
            Tuple of (predicted_letter, confidence_level, reasoning_dict)
        """
        response = response.strip()
        reasoning_dict = {}
        
        # 1. Attempt to parse as JSON to extract structured reasoning
        try:
            data = json.loads(response)
            
            # Extract reasoning components
            reasoning_dict = {
                'question_analysis': data.get('প্রশ্ন_বিশ্লেষণ', data.get('question_analysis', '')),
                #'key_words': data.get('মূল_শব্দ', data.get('key_words', [])),
                #'option_evaluation': data.get('বিকল্প_মূল্যায়ন', data.get('option_evaluation', {})),
                'logic': data.get('যুক্তি', data.get('logic', '')),
                'structured_response': True
            }
            
            # Extract answer
            predicted_option = data.get('উত্তর', data.get('answer', '')).strip().upper()
            if re.match(r'^[ABCD]$', predicted_option):
                return predicted_option, "high", reasoning_dict
                
        except (json.JSONDecodeError, AttributeError):
            reasoning_dict['structured_response'] = False
            reasoning_dict['raw_text'] = response
        
        # 2. Fallback to regex-based extraction (same as zero-shot)
        
        # Look for "উত্তর": "X" pattern in JSON-like text
        json_answer_match = re.search(r'"(?:উত্তর|answer)"\s*:\s*"([ABCD])"', response, re.IGNORECASE)
        if json_answer_match:
            return json_answer_match.group(1).upper(), "medium", reasoning_dict
        
        # Look for single letter answers (A, B, C, D)
        single_letter_match = re.search(r'^([ABCD])$', response.upper(), re.MULTILINE)
        if single_letter_match:
            return single_letter_match.group(1), "low", reasoning_dict
        
        # Look for "উত্তর: X" or "Answer: X" pattern
        answer_pattern = re.search(r'(?:উত্তর|Answer)[:\s]*([ABCD])', response, re.IGNORECASE)
        if answer_pattern:
            return answer_pattern.group(1).upper(), "medium", reasoning_dict
        
        # Look for any A, B, C, or D in the response
        letter_match = re.search(r'([ABCD])', response.upper())
        if letter_match:
            return letter_match.group(1), "low", reasoning_dict
        
        # If no clear answer found
        return "UNKNOWN", "none", reasoning_dict
    
    def process_riddle(self, delay: int, riddle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single riddle with CoT and return result"""
        riddle_id = riddle_data['id']
        
        # Generate CoT prompt
        prompt = self.generate_cot_prompt(riddle_data)
        
        # Wait to avoid rate limits
        time.sleep(delay)
        
        # Get model response
        raw_response = self.call_model(prompt)
        
        # Extract prediction and reasoning
        predicted_letter, confidence, reasoning = self.extract_answer_from_cot(raw_response)
        
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
            'reasoning': reasoning,
            'reasoning_type': riddle_data['reasoning_type'],
            'answer_type': riddle_data['answer_type'],
            'difficulty': riddle_data['difficulty'],
            'trap_type': riddle_data['trap_type'],
            'cultural_depth': riddle_data['cultural_depth'],
            'options': riddle_data['options'],
            'distractors': riddle_data['distractors'],
            'provider': self.provider,
            'model': self.model,
            'prompt_type': 'chain_of_thought',
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def save_results(self, append_mode: bool = False):
        """Save results to file"""
        # Create metadata
        metadata = {
            'provider': self.provider,
            'model': self.model,
            'prompt_type': 'chain_of_thought',
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
        
        # Accuracy by confidence level
        confidence_accuracy = {}
        for confidence in set(r['confidence'] for r in self.results):
            confidence_results = [r for r in self.results if r['confidence'] == confidence]
            confidence_correct = sum(1 for r in confidence_results if r['is_correct'])
            confidence_accuracy[confidence] = confidence_correct / len(confidence_results)
        
        return {
            'overall': overall_accuracy,
            'by_reasoning_type': reasoning_accuracy,
            'by_difficulty': difficulty_accuracy,
            'by_confidence': confidence_accuracy,
            'total_processed': total,
            'total_correct': correct
        }
    
    def run_evaluation(self, delay=5, batch_delay=10, start_index=0):
        """Run the complete CoT evaluation"""
        print(f"Starting Chain of Thought evaluation...")
        print(f"Provider: {self.provider}")
        print(f"Model: {self.model}")
        print(f"Prompt Type: Chain of Thought")
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
            
            # Process riddle with CoT
            result = self.process_riddle(delay, riddle)
            self.results.append(result)
            self.processed_count += 1
            
            # Print immediate result
            status = "✓" if result['is_correct'] else "✗"
            structured = "📋" if result['reasoning'].get('structured_response', False) else "📝"
            print(f"  {status} {structured} Predicted: {result['predicted_option']}, Correct: {result['correct_option']}")
            
            # Save every 10 riddles
            if (i + 1) % 10 == 0:
                self.save_results()
                accuracy = self.calculate_accuracy()
                print(f"  Checkpoint: {accuracy['total_correct']}/{accuracy['total_processed']} correct ({accuracy['overall']:.2%})")
                print(f"  Batch delay: {batch_delay}s")
                time.sleep(batch_delay)
                print()
        
        # Final save
        self.save_results()
        
        # Print final statistics
        print("\n" + "=" * 50)
        print("CHAIN OF THOUGHT EVALUATION COMPLETE!")
        print("=" * 50)
        
        final_accuracy = self.calculate_accuracy()
        print(f"Overall Accuracy: {final_accuracy['overall']:.2%} ({final_accuracy['total_correct']}/{final_accuracy['total_processed']})")
        
        print("\nAccuracy by Reasoning Type:")
        for reasoning_type, acc in final_accuracy['by_reasoning_type'].items():
            print(f"  {reasoning_type}: {acc:.2%}")
        
        print("\nAccuracy by Difficulty:")
        for difficulty, acc in final_accuracy['by_difficulty'].items():
            print(f"  {difficulty}: {acc:.2%}")
        
        print("\nAccuracy by Confidence:")
        for confidence, acc in final_accuracy['by_confidence'].items():
            print(f"  {confidence}: {acc:.2%}")
        
        # Count structured responses
        structured_count = sum(1 for r in self.results if r['reasoning'].get('structured_response', False))
        print(f"\nStructured Responses: {structured_count}/{len(self.results)} ({structured_count/len(self.results):.1%})")
        
        print(f"\nResults saved to: {self.results_file}")

def main():
    """Main function to run CoT evaluation on multiple models"""
    
    settings = [

        # {
        #     'provider': 'novita',
        #     'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
        #     'delay': 6,
        #     'batch_delay': 15
        # }
        # ,
        # {
        #     'provider': 'novita',
        #     'model': 'meta-llama/llama-4-maverick-17b-128e-instruct-fp8',
        #     'delay': 6,
        #     'batch_delay': 15
        # }
        
        # ,

        {
            'provider': 'novita',
            'model': 'qwen/qwen3-235b-a22b-instruct-2507',
            'delay': 6,
            'batch_delay': 15
        }
        ,
        # {
        #     'provider': 'deepseek',
        #     'model': 'deepseek-chat',
        #     'delay': 6,
        #     'batch_delay': 15
        # }
        # ,
        #  
        
        {
            'provider': 'anthropic',
            'model': 'claude-sonnet-4-0',
            'delay': 6,
            'batch_delay': 15
        },
        {
            'provider': 'anthropic',
            'model': 'claude-opus-4-1',
            'delay': 6,
            'batch_delay': 15
        }
        ,
        {
            'provider': 'openai',
            'model': 'gpt-4.1',
            'delay': 6,
            'batch_delay': 15
        }
        ,
        {
            'provider': 'openai',
            'model': 'gpt-5',
            'delay': 6,
            'batch_delay': 15
        }
    ]


    
    data_file = "results/hardest_data_points.json"  # Focus on hard cases
    output_dir = "results/chain_of_thought_hard_cases"
    
    for setting in settings:
        provider = setting['provider']
        model = setting['model']
        
        print(f"\n{'='*60}")
        print(f"Starting evaluation for {provider}/{model}")
        print(f"{'='*60}")
        
        evaluator = CoTRiddleEvaluator(
            provider=provider,
            model=model,
            data_file=data_file,
            output_dir=output_dir
        )
        
        start_index = 0  # Set to continue from specific riddle if needed
        evaluator.run_evaluation(
            delay=setting['delay'],
            batch_delay=setting['batch_delay'],
            start_index=start_index
        )
        
        print(f"\nCompleted evaluation for {provider}/{model}")

if __name__ == "__main__":
    main()