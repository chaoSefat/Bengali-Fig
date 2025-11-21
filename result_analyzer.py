#!/usr/bin/env python3
"""
Results Analysis Script for Bengali Riddle Evaluation
Analyzes saved results and generates comprehensive metrics
"""

import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter
import pandas as pd

class ResultsAnalyzer:
    def __init__(self, results_files: List[str]):
        """
        Initialize analyzer with result files
        
        Args:
            results_files: List of paths to result JSON files
        """
        self.results_files = results_files
        self.all_results = []
        self.load_all_results()
    
    def load_all_results(self):
        """Load all results from files"""
        for file_path in self.results_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Add model info to each result
                for result in data['results']:
                    result['file_metadata'] = data['metadata']
                self.all_results.extend(data['results'])
        
        print(f"Loaded {len(self.all_results)} total results from {len(self.results_files)} files")
    
    def calculate_primary_metrics(self) -> Dict[str, Any]:
        """Calculate primary evaluation metrics"""
        metrics = {}
        
        # Overall accuracy by model
        model_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        # Reasoning type analysis
        reasoning_analysis = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
        
        # Difficulty analysis
        difficulty_analysis = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
        
        # Cultural depth analysis
        cultural_analysis = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
        
        for result in self.all_results:
            model_id = f"{result['provider']}_{result['model']}"
            
            # Overall accuracy
            model_accuracy[model_id]['total'] += 1
            if result['is_correct']:
                model_accuracy[model_id]['correct'] += 1
            
            # Reasoning type accuracy
            reasoning_analysis[model_id][result['reasoning_type']]['total'] += 1
            if result['is_correct']:
                reasoning_analysis[model_id][result['reasoning_type']]['correct'] += 1
            
            # Difficulty accuracy
            difficulty_analysis[model_id][result['difficulty']]['total'] += 1
            if result['is_correct']:
                difficulty_analysis[model_id][result['difficulty']]['correct'] += 1
            
            # Cultural depth accuracy
            cultural_analysis[model_id][result['cultural_depth']]['total'] += 1
            if result['is_correct']:
                cultural_analysis[model_id][result['cultural_depth']]['correct'] += 1
        
        # Convert to percentages
        metrics['overall_accuracy'] = {}
        for model_id, stats in model_accuracy.items():
            metrics['overall_accuracy'][model_id] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        metrics['reasoning_type_accuracy'] = {}
        for model_id, reasoning_stats in reasoning_analysis.items():
            metrics['reasoning_type_accuracy'][model_id] = {}
            for reasoning_type, stats in reasoning_stats.items():
                metrics['reasoning_type_accuracy'][model_id][reasoning_type] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        metrics['difficulty_accuracy'] = {}
        for model_id, difficulty_stats in difficulty_analysis.items():
            metrics['difficulty_accuracy'][model_id] = {}
            for difficulty, stats in difficulty_stats.items():
                metrics['difficulty_accuracy'][model_id][difficulty] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        metrics['cultural_depth_accuracy'] = {}
        for model_id, cultural_stats in cultural_analysis.items():
            metrics['cultural_depth_accuracy'][model_id] = {}
            for cultural_depth, stats in cultural_stats.items():
                metrics['cultural_depth_accuracy'][model_id][cultural_depth] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        return metrics
    
    def calculate_secondary_metrics(self) -> Dict[str, Any]:
        """Calculate secondary evaluation metrics"""
        metrics = {}
        
        # Trap susceptibility analysis
        trap_analysis = defaultdict(lambda: defaultdict(lambda: {'fell_for_trap': 0, 'total': 0}))
        
        # Distractor analysis
        distractor_confusion = defaultdict(lambda: defaultdict(int))
        
        # Confidence analysis
        confidence_analysis = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
        
        for result in self.all_results:
            model_id = f"{result['provider']}_{result['model']}"
            
            # Trap susceptibility (when model gets it wrong due to surface literal interpretation)
            if result['trap_type'] == 'surface_literal':
                trap_analysis[model_id][result['trap_type']]['total'] += 1
                if not result['is_correct']:
                    trap_analysis[model_id][result['trap_type']]['fell_for_trap'] += 1
            
            # Distractor analysis (which wrong answers confuse models most)
            if not result['is_correct'] and result['predicted_option'] != 'UNKNOWN':
                # Find which distractor was chosen
                try:
                    option_index = ['A', 'B', 'C', 'D'].index(result['predicted_option'])
                    if option_index < len(result['options']):
                        chosen_distractor = result['options'][option_index]
                        distractor_confusion[model_id][chosen_distractor] += 1
                except (ValueError, IndexError):
                    pass
            
            # Confidence analysis
            confidence_analysis[model_id][result['confidence']]['total'] += 1
            if result['is_correct']:
                confidence_analysis[model_id][result['confidence']]['correct'] += 1
        
        # Convert trap analysis to percentages
        metrics['trap_susceptibility'] = {}
        for model_id, trap_stats in trap_analysis.items():
            metrics['trap_susceptibility'][model_id] = {}
            for trap_type, stats in trap_stats.items():
                metrics['trap_susceptibility'][model_id][trap_type] = stats['fell_for_trap'] / stats['total'] if stats['total'] > 0 else 0
        
        # Top confused distractors per model
        metrics['top_confusing_distractors'] = {}
        for model_id, distractors in distractor_confusion.items():
            top_distractors = Counter(distractors).most_common(10)
            metrics['top_confusing_distractors'][model_id] = top_distractors
        
        # Confidence calibration
        metrics['confidence_calibration'] = {}
        for model_id, confidence_stats in confidence_analysis.items():
            metrics['confidence_calibration'][model_id] = {}
            for confidence_level, stats in confidence_stats.items():
                metrics['confidence_calibration'][model_id][confidence_level] = {
                    'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                    'count': stats['total']
                }
        
        return metrics
    
    def generate_comparison_table(self, metrics: Dict[str, Any]) -> pd.DataFrame:
        """Generate comparison table for models"""
        models = list(metrics['overall_accuracy'].keys())
        
        comparison_data = []
        for model in models:
            row = {'Model': model}
            
            # Overall accuracy
            row['Overall_Accuracy'] = f"{metrics['overall_accuracy'][model]:.3f}"
            
            # Reasoning type accuracies
            for reasoning_type in ['wordplay', 'metaphor', 'cultural_inference', 'logical']:
                if model in metrics['reasoning_type_accuracy'] and reasoning_type in metrics['reasoning_type_accuracy'][model]:
                    row[f'{reasoning_type}_acc'] = f"{metrics['reasoning_type_accuracy'][model][reasoning_type]:.3f}"
                else:
                    row[f'{reasoning_type}_acc'] = "N/A"
            
            # Difficulty accuracies
            for difficulty in ['easy', 'medium', 'hard']:
                if model in metrics['difficulty_accuracy'] and difficulty in metrics['difficulty_accuracy'][model]:
                    row[f'{difficulty}_acc'] = f"{metrics['difficulty_accuracy'][model][difficulty]:.3f}"
                else:
                    row[f'{difficulty}_acc'] = "N/A"
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def print_detailed_analysis(self):
        """Print comprehensive analysis"""
        print("=" * 80)
        print("BENGALI RIDDLE EVALUATION ANALYSIS")
        print("=" * 80)
        
        # Calculate metrics
        primary_metrics = self.calculate_primary_metrics()
        secondary_metrics = self.calculate_secondary_metrics()
        
        # Overall performance ranking
        print("\n1. OVERALL PERFORMANCE RANKING")
        print("-" * 40)
        overall_ranking = sorted(primary_metrics['overall_accuracy'].items(), 
                                key=lambda x: x[1], reverse=True)
        
        for i, (model, accuracy) in enumerate(overall_ranking, 1):
            print(f"{i:2d}. {model:<30} {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Reasoning type analysis
        print("\n2. PERFORMANCE BY REASONING TYPE")
        print("-" * 40)
        reasoning_types = set()
        for model_stats in primary_metrics['reasoning_type_accuracy'].values():
            reasoning_types.update(model_stats.keys())
        
        for reasoning_type in sorted(reasoning_types):
            print(f"\n{reasoning_type.upper()}:")
            model_scores = []
            for model, stats in primary_metrics['reasoning_type_accuracy'].items():
                if reasoning_type in stats:
                    model_scores.append((model, stats[reasoning_type]))
            
            model_scores.sort(key=lambda x: x[1], reverse=True)
            for model, score in model_scores:
                print(f"  {model:<30} {score:.3f}")
        
        # Difficulty analysis
        print("\n3. PERFORMANCE BY DIFFICULTY")
        print("-" * 40)
        difficulties = ['easy', 'medium', 'hard']
        
        for difficulty in difficulties:
            print(f"\n{difficulty.upper()}:")
            model_scores = []
            for model, stats in primary_metrics['difficulty_accuracy'].items():
                if difficulty in stats:
                    model_scores.append((model, stats[difficulty]))
            
            model_scores.sort(key=lambda x: x[1], reverse=True)
            for model, score in model_scores:
                print(f"  {model:<30} {score:.3f}")
        
        # Cultural depth analysis
        print("\n4. PERFORMANCE BY CULTURAL DEPTH")
        print("-" * 40)
        cultural_depths = set()
        for model_stats in primary_metrics['cultural_depth_accuracy'].values():
            cultural_depths.update(model_stats.keys())
        
        for cultural_depth in sorted(cultural_depths):
            print(f"\n{cultural_depth.upper()}:")
            model_scores = []
            for model, stats in primary_metrics['cultural_depth_accuracy'].items():
                if cultural_depth in stats:
                    model_scores.append((model, stats[cultural_depth]))
            
            model_scores.sort(key=lambda x: x[1], reverse=True)
            for model, score in model_scores:
                print(f"  {model:<30} {score:.3f}")
        
        # Trap susceptibility
        print("\n5. TRAP SUSCEPTIBILITY ANALYSIS")
        print("-" * 40)
        if secondary_metrics['trap_susceptibility']:
            for model, trap_stats in secondary_metrics['trap_susceptibility'].items():
                for trap_type, susceptibility in trap_stats.items():
                    print(f"{model:<30} {trap_type}: {susceptibility:.3f} ({susceptibility*100:.1f}% fell for trap)")
        else:
            print("No trap susceptibility data available")
        
        # Top confusing distractors
        print("\n6. MOST CONFUSING DISTRACTORS")
        print("-" * 40)
        for model, distractors in secondary_metrics['top_confusing_distractors'].items():
            print(f"\n{model}:")
            for distractor, count in distractors[:5]:  # Top 5
                print(f"  '{distractor}' - chosen incorrectly {count} times")
        
        # Generate CSV export
        comparison_df = self.generate_comparison_table(primary_metrics)
        csv_path = "result_analysis/zero_shot_model_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        print(f"\n7. RESULTS EXPORTED")
        print("-" * 40)
        print(f"Detailed comparison table saved to: {csv_path}")
        
        return primary_metrics, secondary_metrics

def main():

    
    results_dir = 'results/zero_shot'  # Directory containing result JSON files

    # Use glob to find all files ending with .json in the specified directory.
    # The sorted() call ensures a consistent processing order.
    results_files = sorted(glob.glob(f"{results_dir}/*.json"))

    if not results_files:
        print(f"No JSON files found in {results_dir}")
        return

    # Create analyzer and run analysis
    analyzer = ResultsAnalyzer(results_files)
    analyzer.print_detailed_analysis()


    # """Main function"""
    # parser = argparse.ArgumentParser(description="Analyze Bengali Riddle evaluation results")
    
    # parser.add_argument("--results", 
    #                    nargs='+',
    #                    required=True,
    #                    help="Paths to result JSON files")
    
    # args = parser.parse_args()
    
    # # Create analyzer and run analysis
    # analyzer = ResultsAnalyzer(args.results)
    # analyzer.print_detailed_analysis()

if __name__ == "__main__":
    main()