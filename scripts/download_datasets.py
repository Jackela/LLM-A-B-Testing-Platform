#!/usr/bin/env python3
"""
Comprehensive Dataset Downloader for LLM A/B Testing Platform

This script downloads, processes, and standardizes benchmark datasets for LLM evaluation.
Designed with robust error handling, resume functionality, and quality assurance.
"""

import asyncio
import gzip
import hashlib
import json
import logging
import shutil
import sys
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urlparse

import aiohttp
import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dataset_downloader.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Standardized dataset information structure"""
    name: str
    description: str
    source_type: str  # 'huggingface', 'github', 'direct_url', 'api'
    source_identifier: str
    download_url: Optional[str] = None
    expected_samples: Optional[int] = None
    languages: List[str] = None
    format_type: str = "json"
    category: str = "general"
    difficulty_levels: List[str] = None
    requires_processing: bool = True
    checksum: Optional[str] = None


@dataclass 
class ProcessedSample:
    """Standardized sample format for all datasets"""
    id: str
    prompt: str
    expected_output: Optional[str] = None
    ground_truth: Optional[str] = None
    category: str = "general"
    difficulty: str = "medium"
    source: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DatasetDownloader:
    """Comprehensive dataset downloader with error handling and validation"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.logs_dir = Path("logs")
        
        # Create directories
        for directory in [self.base_dir, self.raw_dir, self.processed_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True, parents=True)
        
        # Dataset catalog
        self.dataset_catalog = self._build_dataset_catalog()
        
        # Download session
        self.session = None
        
    def _build_dataset_catalog(self) -> Dict[str, DatasetInfo]:
        """Build comprehensive dataset catalog"""
        return {
            # Core Benchmark Datasets
            "mmlu": DatasetInfo(
                name="MMLU",
                description="Massive Multitask Language Understanding across 57 academic subjects",
                source_type="huggingface",
                source_identifier="cais/mmlu",
                expected_samples=15908,
                languages=["en"],
                category="knowledge",
                difficulty_levels=["easy", "medium", "hard"]
            ),
            
            "hellaswag": DatasetInfo(
                name="HellaSwag",
                description="Commonsense natural language inference through sentence completion",
                source_type="huggingface", 
                source_identifier="Rowan/hellaswag",
                expected_samples=70000,
                languages=["en"],
                category="reasoning",
                difficulty_levels=["medium", "hard"]
            ),
            
            "truthfulqa": DatasetInfo(
                name="TruthfulQA",
                description="Benchmark for truthful answer generation",
                source_type="huggingface",
                source_identifier="truthful_qa",
                expected_samples=817,
                languages=["en"],
                category="safety",
                difficulty_levels=["hard"]
            ),
            
            # Specialized Capability Datasets
            "gsm8k": DatasetInfo(
                name="GSM8K",
                description="Grade School Math word problems requiring multi-step reasoning",
                source_type="huggingface",
                source_identifier="gsm8k",
                expected_samples=8500,
                languages=["en"],
                category="mathematics",
                difficulty_levels=["easy", "medium"]
            ),
            
            "human_eval": DatasetInfo(
                name="HumanEval",
                description="Hand-written programming problems for code generation evaluation",
                source_type="huggingface",
                source_identifier="openai_humaneval",
                expected_samples=164,
                languages=["en"],
                category="coding",
                difficulty_levels=["medium", "hard"]
            ),
            
            "race": DatasetInfo(
                name="RACE",
                description="Reading comprehension from examinations",
                source_type="huggingface",
                source_identifier="race",
                expected_samples=27933,
                languages=["en"],
                category="comprehension",
                difficulty_levels=["easy", "medium", "hard"]
            ),
            
            "ai2_arc": DatasetInfo(
                name="ARC",
                description="AI2 Reasoning Challenge with grade-school science questions",
                source_type="huggingface",
                source_identifier="ai2_arc",
                expected_samples=7787,
                languages=["en"],
                category="science",
                difficulty_levels=["easy", "hard"]
            ),
            
            # Creative and Language Generation
            "writing_prompts": DatasetInfo(
                name="WritingPrompts",
                description="Creative writing prompts with human responses",
                source_type="huggingface",
                source_identifier="writing_prompts",
                expected_samples=300000,
                languages=["en"],
                category="creative",
                difficulty_levels=["medium"]
            ),
            
            "xsum": DatasetInfo(
                name="XSum",
                description="BBC articles with abstractive single-sentence summaries",
                source_type="huggingface",
                source_identifier="xsum",
                expected_samples=227000,
                languages=["en"],
                category="summarization", 
                difficulty_levels=["medium", "hard"]
            ),
            
            # Safety and Bias Evaluation
            "stereoset": DatasetInfo(
                name="StereoSet",
                description="Measuring stereotypical biases across profession, gender, religion, race",
                source_type="huggingface",
                source_identifier="stereoset",
                expected_samples=17000,
                languages=["en"],
                category="bias",
                difficulty_levels=["hard"]
            ),
            
            "wino_bias": DatasetInfo(
                name="WinoBias", 
                description="Gender bias evaluation in coreference resolution",
                source_type="huggingface",
                source_identifier="wino_bias",
                expected_samples=3160,
                languages=["en"],
                category="bias",
                difficulty_levels=["medium"]
            ),
            
            # Multilingual Datasets
            "xnli": DatasetInfo(
                name="XNLI",
                description="Cross-lingual Natural Language Inference in 15 languages",
                source_type="huggingface",
                source_identifier="xnli",
                expected_samples=7500,
                languages=["en", "zh", "es", "ar", "fr", "de", "ru", "th", "tr", "bg", "el", "hi", "sw", "ur", "vi"],
                category="multilingual",
                difficulty_levels=["medium", "hard"]
            ),
            
            "ceval": DatasetInfo(
                name="C-Eval",
                description="Comprehensive Chinese evaluation suite across 52 subjects",
                source_type="huggingface",
                source_identifier="ceval/ceval-exam", 
                expected_samples=13948,
                languages=["zh"],
                category="knowledge",
                difficulty_levels=["easy", "medium", "hard"]
            )
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3600),  # 1 hour timeout
            connector=aiohttp.TCPConnector(limit=10)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def download_huggingface_dataset(self, dataset_info: DatasetInfo) -> bool:
        """Download dataset from Hugging Face"""
        try:
            logger.info(f"Downloading {dataset_info.name} from Hugging Face: {dataset_info.source_identifier}")
            
            # Download with progress tracking
            dataset = load_dataset(dataset_info.source_identifier, trust_remote_code=True)
            
            # Save raw dataset
            raw_path = self.raw_dir / f"{dataset_info.name.lower()}"
            raw_path.mkdir(exist_ok=True)
            
            # Save each split
            total_samples = 0
            for split_name, split_data in dataset.items():
                split_file = raw_path / f"{split_name}.json"
                
                # Convert to list of dictionaries and save
                data_list = []
                for item in tqdm(split_data, desc=f"Processing {split_name}"):
                    data_list.append(dict(item))
                    
                with open(split_file, 'w', encoding='utf-8') as f:
                    json.dump(data_list, f, indent=2, ensure_ascii=False)
                
                total_samples += len(data_list)
                logger.info(f"Saved {len(data_list)} samples to {split_file}")
            
            # Validate sample count
            if dataset_info.expected_samples and abs(total_samples - dataset_info.expected_samples) > dataset_info.expected_samples * 0.1:
                logger.warning(f"Sample count mismatch for {dataset_info.name}: expected {dataset_info.expected_samples}, got {total_samples}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_info.name}: {e}")
            return False
    
    async def download_direct_url(self, dataset_info: DatasetInfo) -> bool:
        """Download dataset from direct URL"""
        if not dataset_info.download_url:
            logger.error(f"No download URL provided for {dataset_info.name}")
            return False
        
        try:
            logger.info(f"Downloading {dataset_info.name} from URL: {dataset_info.download_url}")
            
            async with self.session.get(dataset_info.download_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to download {dataset_info.name}: HTTP {response.status}")
                    return False
                
                content = await response.read()
                
                # Determine filename from URL
                parsed_url = urlparse(dataset_info.download_url)
                filename = Path(parsed_url.path).name or f"{dataset_info.name.lower()}.zip"
                
                raw_file_path = self.raw_dir / filename
                
                with open(raw_file_path, 'wb') as f:
                    f.write(content)
                
                # Extract if archive
                if filename.endswith(('.zip', '.tar.gz', '.gz')):
                    await self._extract_archive(raw_file_path, self.raw_dir / dataset_info.name.lower())
                
                logger.info(f"Successfully downloaded {dataset_info.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to download {dataset_info.name}: {e}")
            return False
    
    async def _extract_archive(self, archive_path: Path, extract_path: Path) -> bool:
        """Extract archive files"""
        try:
            extract_path.mkdir(exist_ok=True, parents=True)
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_file:
                    zip_file.extractall(extract_path)
            elif archive_path.suffix == '.gz':
                if archive_path.name.endswith('.tar.gz'):
                    import tarfile
                    with tarfile.open(archive_path, 'r:gz') as tar_file:
                        tar_file.extractall(extract_path)
                else:
                    with gzip.open(archive_path, 'rb') as gz_file:
                        with open(extract_path / archive_path.stem, 'wb') as out_file:
                            shutil.copyfileobj(gz_file, out_file)
            
            logger.info(f"Extracted {archive_path} to {extract_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract {archive_path}: {e}")
            return False
    
    def process_dataset(self, dataset_name: str, dataset_info: DatasetInfo) -> bool:
        """Process raw dataset into standardized format"""
        try:
            logger.info(f"Processing {dataset_info.name} into standardized format")
            
            raw_path = self.raw_dir / dataset_name.lower()
            processed_path = self.processed_dir / dataset_info.category
            processed_path.mkdir(exist_ok=True, parents=True)
            
            processed_samples = []
            
            # Dataset-specific processing logic
            if dataset_name == "mmlu":
                processed_samples = self._process_mmlu(raw_path)
            elif dataset_name == "hellaswag":
                processed_samples = self._process_hellaswag(raw_path)
            elif dataset_name == "truthfulqa":
                processed_samples = self._process_truthfulqa(raw_path)
            elif dataset_name == "gsm8k":
                processed_samples = self._process_gsm8k(raw_path)
            elif dataset_name == "human_eval":
                processed_samples = self._process_human_eval(raw_path)
            elif dataset_name == "race":
                processed_samples = self._process_race(raw_path)
            elif dataset_name == "ai2_arc":
                processed_samples = self._process_arc(raw_path)
            elif dataset_name == "writing_prompts":
                processed_samples = self._process_writing_prompts(raw_path)
            elif dataset_name == "xsum":
                processed_samples = self._process_xsum(raw_path)
            elif dataset_name == "stereoset":
                processed_samples = self._process_stereoset(raw_path)
            elif dataset_name == "wino_bias":
                processed_samples = self._process_wino_bias(raw_path)
            elif dataset_name == "xnli":
                processed_samples = self._process_xnli(raw_path)
            elif dataset_name == "ceval":
                processed_samples = self._process_ceval(raw_path)
            else:
                logger.warning(f"No specific processor for {dataset_name}, using generic processor")
                processed_samples = self._process_generic(raw_path)
            
            # Save processed dataset
            output_file = processed_path / f"{dataset_name.lower()}.json"
            processed_data = [asdict(sample) for sample in processed_samples]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processed {len(processed_samples)} samples for {dataset_info.name}")
            logger.info(f"Saved processed dataset to {output_file}")
            
            # Generate dataset summary
            self._generate_dataset_summary(dataset_name, dataset_info, processed_samples, output_file)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {dataset_info.name}: {e}")
            return False
    
    def _process_mmlu(self, raw_path: Path) -> List[ProcessedSample]:
        """Process MMLU dataset"""
        samples = []
        
        for json_file in raw_path.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for idx, item in enumerate(data):
                sample = ProcessedSample(
                    id=f"mmlu_{json_file.stem}_{idx}",
                    prompt=f"Question: {item['question']}\nChoices: {', '.join(item['choices'])}",
                    expected_output=item['choices'][item['answer']],
                    ground_truth=item['choices'][item['answer']],
                    category=item.get('subject', 'general'),
                    difficulty="medium",
                    source="MMLU",
                    metadata={
                        "subject": item.get('subject'),
                        "choices": item['choices'],
                        "answer_index": item['answer']
                    }
                )
                samples.append(sample)
        
        return samples
    
    def _process_hellaswag(self, raw_path: Path) -> List[ProcessedSample]:
        """Process HellaSwag dataset"""
        samples = []
        
        for json_file in raw_path.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for idx, item in enumerate(data):
                # Create context and choices
                context = item['ctx']
                choices = item['endings']
                correct_choice = choices[int(item['label'])] if 'label' in item else None
                
                sample = ProcessedSample(
                    id=f"hellaswag_{json_file.stem}_{idx}",
                    prompt=f"Context: {context}\nChoose the most logical continuation:\n" + 
                           "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)]),
                    expected_output=correct_choice,
                    ground_truth=correct_choice,
                    category="commonsense",
                    difficulty="medium",
                    source="HellaSwag",
                    metadata={
                        "context": context,
                        "choices": choices,
                        "label": item.get('label'),
                        "activity_label": item.get('activity_label')
                    }
                )
                samples.append(sample)
        
        return samples
    
    def _process_truthfulqa(self, raw_path: Path) -> List[ProcessedSample]:
        """Process TruthfulQA dataset"""
        samples = []
        
        for json_file in raw_path.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for idx, item in enumerate(data):
                sample = ProcessedSample(
                    id=f"truthfulqa_{idx}",
                    prompt=item['question'],
                    expected_output=item.get('best_answer', ''),
                    ground_truth=item.get('best_answer', ''),
                    category="truthfulness",
                    difficulty="hard",
                    source="TruthfulQA",
                    metadata={
                        "type": item.get('type'),
                        "category": item.get('category'),
                        "correct_answers": item.get('correct_answers', []),
                        "incorrect_answers": item.get('incorrect_answers', [])
                    }
                )
                samples.append(sample)
        
        return samples
    
    def _process_gsm8k(self, raw_path: Path) -> List[ProcessedSample]:
        """Process GSM8K dataset"""
        samples = []
        
        for json_file in raw_path.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for idx, item in enumerate(data):
                sample = ProcessedSample(
                    id=f"gsm8k_{json_file.stem}_{idx}",
                    prompt=item['question'],
                    expected_output=item['answer'].split('####')[-1].strip(),
                    ground_truth=item['answer'].split('####')[-1].strip(),
                    category="mathematics",
                    difficulty="medium",
                    source="GSM8K",
                    metadata={
                        "full_answer": item['answer'],
                        "solution_steps": item['answer'].split('####')[0].strip()
                    }
                )
                samples.append(sample)
        
        return samples
    
    def _process_human_eval(self, raw_path: Path) -> List[ProcessedSample]:
        """Process HumanEval dataset"""
        samples = []
        
        for json_file in raw_path.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for idx, item in enumerate(data):
                sample = ProcessedSample(
                    id=f"humaneval_{item.get('task_id', idx)}",
                    prompt=f"Complete this Python function:\n\n{item['prompt']}",
                    expected_output=item.get('canonical_solution', ''),
                    ground_truth=item.get('canonical_solution', ''),
                    category="programming",
                    difficulty="hard",
                    source="HumanEval",
                    metadata={
                        "task_id": item.get('task_id'),
                        "entry_point": item.get('entry_point'),
                        "test": item.get('test'),
                        "docstring": item['prompt'].split('"""')[1] if '"""' in item['prompt'] else ''
                    }
                )
                samples.append(sample)
        
        return samples
    
    def _process_race(self, raw_path: Path) -> List[ProcessedSample]:
        """Process RACE dataset"""
        samples = []
        
        for json_file in raw_path.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for idx, item in enumerate(data):
                # RACE has multiple questions per article
                article = item['article']
                for q_idx, (question, options, answer) in enumerate(zip(
                    item['questions'], item['options'], item['answers']
                )):
                    sample = ProcessedSample(
                        id=f"race_{json_file.stem}_{idx}_{q_idx}",
                        prompt=f"Article: {article}\n\nQuestion: {question}\nOptions: {', '.join(options)}",
                        expected_output=options[ord(answer) - ord('A')],
                        ground_truth=options[ord(answer) - ord('A')],
                        category="reading_comprehension",
                        difficulty="medium",
                        source="RACE",
                        metadata={
                            "article": article,
                            "question": question,
                            "options": options,
                            "answer_letter": answer
                        }
                    )
                    samples.append(sample)
        
        return samples
    
    def _process_arc(self, raw_path: Path) -> List[ProcessedSample]:
        """Process ARC dataset"""
        samples = []
        
        for json_file in raw_path.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            difficulty = "hard" if "challenge" in json_file.name.lower() else "easy"
            
            for idx, item in enumerate(data):
                choices = [choice['text'] for choice in item['choices']['choices']]
                correct_idx = ord(item['answerKey']) - ord('A')
                
                sample = ProcessedSample(
                    id=f"arc_{json_file.stem}_{idx}",
                    prompt=f"Question: {item['question']['stem']}\nChoices: {', '.join(choices)}",
                    expected_output=choices[correct_idx],
                    ground_truth=choices[correct_idx],
                    category="science",
                    difficulty=difficulty,
                    source="ARC",
                    metadata={
                        "question_stem": item['question']['stem'],
                        "choices": choices,
                        "answer_key": item['answerKey']
                    }
                )
                samples.append(sample)
        
        return samples
    
    def _process_writing_prompts(self, raw_path: Path) -> List[ProcessedSample]:
        """Process WritingPrompts dataset (sample subset for efficiency)"""
        samples = []
        
        for json_file in raw_path.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Sample only first 1000 entries to avoid memory issues
            for idx, item in enumerate(data[:1000]):
                sample = ProcessedSample(
                    id=f"writing_prompts_{idx}",
                    prompt=item.get('prompt', ''),
                    expected_output=item.get('response', ''),
                    ground_truth=item.get('response', ''),
                    category="creative_writing",
                    difficulty="medium",
                    source="WritingPrompts",
                    metadata={
                        "prompt": item.get('prompt'),
                        "response": item.get('response')
                    }
                )
                samples.append(sample)
        
        return samples
    
    def _process_xsum(self, raw_path: Path) -> List[ProcessedSample]:
        """Process XSum dataset (sample subset)"""
        samples = []
        
        for json_file in raw_path.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Sample only first 1000 entries
            for idx, item in enumerate(data[:1000]):
                sample = ProcessedSample(
                    id=f"xsum_{idx}",
                    prompt=f"Summarize this article in one sentence:\n\n{item.get('document', '')}",
                    expected_output=item.get('summary', ''),
                    ground_truth=item.get('summary', ''),
                    category="summarization",
                    difficulty="hard",
                    source="XSum",
                    metadata={
                        "document": item.get('document'),
                        "summary": item.get('summary')
                    }
                )
                samples.append(sample)
        
        return samples
    
    def _process_stereoset(self, raw_path: Path) -> List[ProcessedSample]:
        """Process StereoSet dataset"""
        samples = []
        
        for json_file in raw_path.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle nested structure
            if 'data' in data:
                entries = data['data']['intrasentence'] + data['data']['intersentence']
            else:
                entries = data
            
            for idx, item in enumerate(entries):
                context = item.get('context', '')
                for sentence in item.get('sentences', []):
                    sample = ProcessedSample(
                        id=f"stereoset_{idx}_{sentence.get('id', '')}",
                        prompt=f"Context: {context}\nComplete: {sentence.get('sentence', '')}",
                        expected_output="[Evaluate for bias]",
                        ground_truth=sentence.get('gold_label', 'unrelated'),
                        category="bias_detection",
                        difficulty="hard",
                        source="StereoSet",
                        metadata={
                            "context": context,
                            "sentence": sentence.get('sentence'),
                            "gold_label": sentence.get('gold_label'),
                            "bias_type": item.get('bias_type')
                        }
                    )
                    samples.append(sample)
        
        return samples
    
    def _process_wino_bias(self, raw_path: Path) -> List[ProcessedSample]:
        """Process WinoBias dataset"""
        samples = []
        
        for json_file in raw_path.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for idx, item in enumerate(data):
                sample = ProcessedSample(
                    id=f"wino_bias_{idx}",
                    prompt=f"Resolve the pronoun in this sentence: {item.get('text', '')}",
                    expected_output=item.get('answer', ''),
                    ground_truth=item.get('answer', ''),
                    category="bias_detection",
                    difficulty="medium",
                    source="WinoBias",
                    metadata={
                        "text": item.get('text'),
                        "answer": item.get('answer'),
                        "bias_type": item.get('bias_type')
                    }
                )
                samples.append(sample)
        
        return samples
    
    def _process_xnli(self, raw_path: Path) -> List[ProcessedSample]:
        """Process XNLI dataset"""
        samples = []
        
        for json_file in raw_path.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for idx, item in enumerate(data):
                label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
                
                sample = ProcessedSample(
                    id=f"xnli_{item.get('language', 'en')}_{idx}",
                    prompt=f"Premise: {item.get('premise', '')}\nHypothesis: {item.get('hypothesis', '')}\nRelationship:",
                    expected_output=label_map.get(item.get('label'), 'neutral'),
                    ground_truth=label_map.get(item.get('label'), 'neutral'),
                    category="natural_language_inference",
                    difficulty="hard",
                    source="XNLI",
                    metadata={
                        "premise": item.get('premise'),
                        "hypothesis": item.get('hypothesis'),
                        "language": item.get('language'),
                        "label": item.get('label')
                    }
                )
                samples.append(sample)
        
        return samples
    
    def _process_ceval(self, raw_path: Path) -> List[ProcessedSample]:
        """Process C-Eval dataset"""
        samples = []
        
        for json_file in raw_path.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for idx, item in enumerate(data):
                choices = [item.get('A', ''), item.get('B', ''), item.get('C', ''), item.get('D', '')]
                correct_choice = choices[ord(item.get('answer', 'A')) - ord('A')]
                
                sample = ProcessedSample(
                    id=f"ceval_{json_file.stem}_{idx}",
                    prompt=f"问题: {item.get('question', '')}\n选择: A) {choices[0]} B) {choices[1]} C) {choices[2]} D) {choices[3]}",
                    expected_output=correct_choice,
                    ground_truth=correct_choice,
                    category=item.get('subject', 'general'),
                    difficulty="medium",
                    source="C-Eval",
                    metadata={
                        "question": item.get('question'),
                        "choices": choices,
                        "answer": item.get('answer'),
                        "subject": item.get('subject')
                    }
                )
                samples.append(sample)
        
        return samples
    
    def _process_generic(self, raw_path: Path) -> List[ProcessedSample]:
        """Generic processor for unknown datasets"""
        samples = []
        
        for json_file in raw_path.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for idx, item in enumerate(data):
                # Try to extract common fields
                prompt = item.get('question', item.get('prompt', item.get('input', str(item))))
                answer = item.get('answer', item.get('output', item.get('target', '')))
                
                sample = ProcessedSample(
                    id=f"generic_{json_file.stem}_{idx}",
                    prompt=prompt,
                    expected_output=answer,
                    ground_truth=answer,
                    category="general",
                    difficulty="medium",
                    source="Generic",
                    metadata=item
                )
                samples.append(sample)
        
        return samples
    
    def _generate_dataset_summary(self, dataset_name: str, dataset_info: DatasetInfo, 
                                samples: List[ProcessedSample], output_file: Path) -> None:
        """Generate comprehensive dataset summary"""
        summary = {
            "dataset_name": dataset_info.name,
            "description": dataset_info.description,
            "source": dataset_info.source_identifier,
            "category": dataset_info.category,
            "languages": dataset_info.languages,
            "total_samples": len(samples),
            "expected_samples": dataset_info.expected_samples,
            "sample_match": abs(len(samples) - (dataset_info.expected_samples or 0)) < (dataset_info.expected_samples or 0) * 0.1,
            "categories": list(set(sample.category for sample in samples)),
            "difficulty_levels": list(set(sample.difficulty for sample in samples)),
            "average_prompt_length": sum(len(sample.prompt) for sample in samples) / len(samples) if samples else 0,
            "output_file": str(output_file),
            "processed_at": datetime.now().isoformat(),
            "sample_preview": [asdict(samples[i]) for i in range(min(3, len(samples)))]
        }
        
        summary_file = output_file.parent / f"{dataset_name.lower()}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated summary for {dataset_info.name}: {summary_file}")
    
    async def download_all_datasets(self, dataset_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """Download and process all specified datasets"""
        if dataset_names is None:
            dataset_names = list(self.dataset_catalog.keys())
        
        results = {}
        
        for name in dataset_names:
            if name not in self.dataset_catalog:
                logger.warning(f"Unknown dataset: {name}")
                results[name] = False
                continue
            
            dataset_info = self.dataset_catalog[name]
            
            try:
                # Download
                if dataset_info.source_type == "huggingface":
                    download_success = self.download_huggingface_dataset(dataset_info)
                elif dataset_info.source_type == "direct_url":
                    download_success = await self.download_direct_url(dataset_info)
                else:
                    logger.warning(f"Unsupported source type for {name}: {dataset_info.source_type}")
                    download_success = False
                
                if not download_success:
                    results[name] = False
                    continue
                
                # Process
                if dataset_info.requires_processing:
                    process_success = self.process_dataset(name, dataset_info)
                    results[name] = process_success
                else:
                    results[name] = True
                    
            except Exception as e:
                logger.error(f"Failed to download/process {name}: {e}")
                results[name] = False
        
        return results
    
    def generate_master_report(self, results: Dict[str, bool]) -> None:
        """Generate master report of all downloads"""
        successful = [name for name, success in results.items() if success]
        failed = [name for name, success in results.items() if not success]
        
        report = {
            "download_summary": {
                "total_datasets": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(results) if results else 0
            },
            "successful_downloads": successful,
            "failed_downloads": failed,
            "dataset_catalog": {name: asdict(info) for name, info in self.dataset_catalog.items()},
            "generated_at": datetime.now().isoformat(),
            "directory_structure": {
                "raw_data": str(self.raw_dir),
                "processed_data": str(self.processed_dir),
                "logs": str(self.logs_dir)
            }
        }
        
        report_file = self.base_dir / "download_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Also generate markdown report
        self._generate_markdown_report(report, self.base_dir / "DATASET_REPORT.md")
        
        logger.info(f"Generated master report: {report_file}")
    
    def _generate_markdown_report(self, report: Dict, output_file: Path) -> None:
        """Generate markdown report"""
        md_content = f"""# Dataset Download Report

Generated: {report['generated_at']}

## Summary
- **Total Datasets**: {report['download_summary']['total_datasets']}
- **Successful**: {report['download_summary']['successful']}  
- **Failed**: {report['download_summary']['failed']}
- **Success Rate**: {report['download_summary']['success_rate']:.1%}

## Directory Structure
- **Raw Data**: `{report['directory_structure']['raw_data']}`
- **Processed Data**: `{report['directory_structure']['processed_data']}`  
- **Logs**: `{report['directory_structure']['logs']}`

## Successful Downloads
"""
        
        for dataset in report['successful_downloads']:
            md_content += f"- ✅ {dataset}\n"
        
        if report['failed_downloads']:
            md_content += "\n## Failed Downloads\n"
            for dataset in report['failed_downloads']:
                md_content += f"- ❌ {dataset}\n"
        
        md_content += "\n## Dataset Catalog\n"
        for name, info in report['dataset_catalog'].items():
            status = "✅" if name in report['successful_downloads'] else "❌"
            md_content += f"\n### {status} {info['name']}\n"
            md_content += f"- **Description**: {info['description']}\n"
            md_content += f"- **Source**: {info['source_identifier']}\n"
            md_content += f"- **Category**: {info['category']}\n"
            md_content += f"- **Languages**: {', '.join(info.get('languages', ['Unknown']))}\n"
            md_content += f"- **Expected Samples**: {info.get('expected_samples', 'Unknown')}\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)


async def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and process LLM evaluation datasets")
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to download')
    parser.add_argument('--data-dir', default='data', help='Base directory for datasets')
    parser.add_argument('--skip-processing', action='store_true', help='Skip processing step')
    
    args = parser.parse_args()
    
    # Ensure logs directory
    Path("logs").mkdir(exist_ok=True)
    
    async with DatasetDownloader(args.data_dir) as downloader:
        logger.info("Starting dataset download process")
        
        results = await downloader.download_all_datasets(args.datasets)
        
        # Generate reports
        downloader.generate_master_report(results)
        
        # Print summary
        successful = sum(results.values())
        total = len(results)
        
        print(f"\n{'='*50}")
        print(f"DOWNLOAD COMPLETE")
        print(f"{'='*50}")
        print(f"Success: {successful}/{total} datasets")
        print(f"Success Rate: {successful/total:.1%}")
        print(f"Data Directory: {downloader.base_dir}")
        print(f"Report: {downloader.base_dir}/DATASET_REPORT.md")
        
        return 0 if successful == total else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)