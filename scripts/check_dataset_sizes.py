#!/usr/bin/env python3
"""
Quick dataset size checker to identify smaller datasets for testing.
"""

from datasets import load_dataset_builder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Small subset of datasets to check
DATASETS_TO_CHECK = {
    "truthfulqa": "truthful_qa/mc",
    "arc": "allenai/ai2_arc/ARC-Easy",
    "gsm8k": "openai/gsm8k/main",
    "human_eval": "openai/humaneval",
    "winogrande": "allenai/winogrande/winogrande_xl",
    "hellaswag": "Rowan/hellaswag",
    "boolq": "super_glue/boolq",
    "piqa": "ybisk/piqa",
}

def check_dataset_size(dataset_name, config_name=None):
    """Check the size of a dataset without downloading it."""
    try:
        if config_name:
            builder = load_dataset_builder(dataset_name, config_name)
        else:
            builder = load_dataset_builder(dataset_name)
        
        # Get info about the dataset
        info = builder.info
        splits_info = info.splits
        
        total_examples = 0
        split_details = {}
        
        for split_name, split_info in splits_info.items():
            num_examples = split_info.num_examples
            total_examples += num_examples
            split_details[split_name] = num_examples
        
        return {
            "name": dataset_name,
            "config": config_name,
            "total_examples": total_examples,
            "splits": split_details,
            "description": info.description[:100] + "..." if info.description else "No description"
        }
    
    except Exception as e:
        logger.error(f"Error checking {dataset_name}: {e}")
        return None

def main():
    """Check sizes of various datasets."""
    print("🔍 Checking dataset sizes...\n")
    
    dataset_sizes = []
    
    for name, config in DATASETS_TO_CHECK.items():
        if "/" in config:
            dataset_name, config_name = config.rsplit("/", 1)
        else:
            dataset_name = config
            config_name = None
        
        print(f"Checking {name}...")
        size_info = check_dataset_size(dataset_name, config_name)
        
        if size_info:
            dataset_sizes.append(size_info)
            print(f"  ✅ {size_info['total_examples']:,} total examples")
        else:
            print(f"  ❌ Failed to check")
        print()
    
    # Sort by size (smallest first)
    dataset_sizes.sort(key=lambda x: x['total_examples'])
    
    print("📊 Dataset sizes (smallest to largest):\n")
    print(f"{'Dataset':<20} {'Total Examples':<15} {'Main Split':<20}")
    print("-" * 60)
    
    for ds in dataset_sizes:
        main_split = max(ds['splits'].items(), key=lambda x: x[1])
        print(f"{ds['name']:<20} {ds['total_examples']:,<14} {main_split[0]} ({main_split[1]:,})")
    
    print("\n🎯 Recommended for testing (smaller datasets):")
    small_datasets = [ds for ds in dataset_sizes if ds['total_examples'] < 10000]
    for ds in small_datasets[:3]:
        print(f"  • {ds['name']}: {ds['total_examples']:,} examples")

if __name__ == "__main__":
    main()