#!/usr/bin/env python3
"""
Download small evaluation datasets for testing the LLM A/B Testing Platform.
"""

import json
import logging
from pathlib import Path
from datasets import load_dataset
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_arc_dataset(output_dir: Path) -> bool:
    """Download ARC dataset (smallest: 5,197 examples)."""
    try:
        logger.info("📥 Downloading ARC dataset...")
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy")
        
        # Convert to our standard format
        processed_data = []
        for split_name, split_data in dataset.items():
            for idx, example in enumerate(split_data):
                processed_example = {
                    "id": f"arc_{split_name}_{idx}",
                    "prompt": f"Question: {example['question']}\nChoices: {', '.join(example['choices']['text'])}",
                    "expected_output": example['choices']['text'][example['choices']['label'].index(example['answerKey'])],
                    "ground_truth": example['answerKey'],
                    "category": "science",
                    "difficulty": "easy",
                    "source": "ARC-Easy",
                    "metadata": {
                        "split": split_name,
                        "question": example['question'],
                        "choices": example['choices'],
                        "answer_key": example['answerKey']
                    }
                }
                processed_data.append(processed_example)
        
        # Save processed dataset
        output_file = output_dir / "arc_easy.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ ARC dataset saved: {len(processed_data)} examples → {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading ARC dataset: {e}")
        return False

def download_gsm8k_dataset(output_dir: Path) -> bool:
    """Download GSM8K dataset (8,792 examples)."""
    try:
        logger.info("📥 Downloading GSM8K dataset...")
        dataset = load_dataset("gsm8k", "main")
        
        # Convert to our standard format
        processed_data = []
        for split_name, split_data in dataset.items():
            for idx, example in enumerate(split_data):
                # Extract numeric answer from the solution
                answer_parts = example['answer'].split('####')
                numeric_answer = answer_parts[-1].strip() if len(answer_parts) > 1 else "No answer found"
                
                processed_example = {
                    "id": f"gsm8k_{split_name}_{idx}",
                    "prompt": f"Solve this math problem step by step:\n{example['question']}",
                    "expected_output": example['answer'],
                    "ground_truth": numeric_answer,
                    "category": "math",
                    "difficulty": "medium",
                    "source": "GSM8K",
                    "metadata": {
                        "split": split_name,
                        "question": example['question'],
                        "full_answer": example['answer'],
                        "numeric_answer": numeric_answer
                    }
                }
                processed_data.append(processed_example)
        
        # Save processed dataset
        output_file = output_dir / "gsm8k.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ GSM8K dataset saved: {len(processed_data)} examples → {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading GSM8K dataset: {e}")
        return False

def create_test_sample(datasets_dir: Path, sample_size: int = 20) -> bool:
    """Create a small sample for quick testing."""
    try:
        logger.info(f"📋 Creating test sample ({sample_size} examples)...")
        
        test_samples = []
        
        # Load ARC data
        arc_file = datasets_dir / "arc_easy.json"
        if arc_file.exists():
            with open(arc_file, "r", encoding="utf-8") as f:
                arc_data = json.load(f)
            test_samples.extend(arc_data[:sample_size//2])
        
        # Load GSM8K data  
        gsm8k_file = datasets_dir / "gsm8k.json"
        if gsm8k_file.exists():
            with open(gsm8k_file, "r", encoding="utf-8") as f:
                gsm8k_data = json.load(f)
            test_samples.extend(gsm8k_data[:sample_size//2])
        
        # Save test sample
        sample_file = datasets_dir / "test_sample.json"
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(test_samples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Test sample created: {len(test_samples)} examples → {sample_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating test sample: {e}")
        return False

def main():
    """Download small datasets for testing."""
    print("📚 Downloading small evaluation datasets for LLM A/B Testing Platform")
    print("=" * 70)
    
    # Setup directories
    base_dir = Path("data")
    datasets_dir = base_dir / "processed"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    # Download datasets
    if download_arc_dataset(datasets_dir):
        success_count += 1
    
    if download_gsm8k_dataset(datasets_dir):
        success_count += 1
    
    # Create test sample
    if create_test_sample(datasets_dir):
        success_count += 1
    
    print(f"\n🎉 Download complete! Successfully processed {success_count}/3 tasks")
    
    # Dataset summary
    print("\n📊 Dataset Summary:")
    print("=" * 30)
    
    for dataset_file in datasets_dir.glob("*.json"):
        try:
            with open(dataset_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"📁 {dataset_file.name}: {len(data):,} examples")
            
            # Show sample
            if data:
                sample = data[0]
                print(f"   Sample ID: {sample['id']}")
                print(f"   Category: {sample['category']}")
                print(f"   Source: {sample['source']}")
                print()
                
        except Exception as e:
            print(f"❌ Error reading {dataset_file.name}: {e}")
    
    print("✅ Ready for functional testing!")

if __name__ == "__main__":
    main()