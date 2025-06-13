from datasets import load_dataset
from typing import Dict, List, Optional
import numpy as np
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BillSumDataProcessor:
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        max_length: int = 2048,
        train_size: float = 0.9,
        seed: int = 42
    ):
        """
        Initialize the BillSum data processor.
        
        Args:
            model_name: Name of the model/tokenizer to use
            max_length: Maximum sequence length for tokenization
            train_size: Proportion of data to use for training
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.max_length = max_length
        self.train_size = train_size
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens if needed
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def load_dataset(self) -> Dict:
        """Load and split the billsum dataset."""
        logger.info("Loading billsum dataset...")
        dataset = load_dataset("billsum")
        
        # Split the training data into train and validation
        train_val = dataset["train"].train_test_split(
            train_size=self.train_size,
            seed=self.seed
        )
        
        return {
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": dataset["test"]
        }
    
    def format_prompt(self, text: str) -> str:
        """Format the input text as a prompt."""
        return f"<s>[INST] Summarize the following legal bill:\n\n{text} [/INST]"
    
    def format_response(self, summary: str) -> str:
        """Format the target summary as a response."""
        return f"{summary} </s>"
    
    def prepare_training_data(self, dataset: Dict) -> Dict:
        """Prepare the dataset for training."""
        logger.info("Preparing training data...")
        
        def format_example(example):
            prompt = self.format_prompt(example["text"])
            response = self.format_response(example["summary"])
            return {
                "text": prompt + response,
                "input_text": example["text"],
                "target_summary": example["summary"]
            }
        
        prepared_dataset = {}
        for split, data in dataset.items():
            prepared_dataset[split] = data.map(
                format_example,
                remove_columns=data.column_names,
                desc=f"Formatting {split} split"
            )
        
        return prepared_dataset
    
    def tokenize_data(self, dataset: Dict) -> Dict:
        """Tokenize the formatted dataset."""
        logger.info("Tokenizing dataset...")
        
        def tokenize_function(example):
            tokenized = self.tokenizer(
                example["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            return {
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0]
            }
        
        tokenized_dataset = {}
        for split, data in dataset.items():
            tokenized_dataset[split] = data.map(
                tokenize_function,
                remove_columns=data.column_names,
                desc=f"Tokenizing {split} split"
            )
        
        return tokenized_dataset
    
    def process(self) -> Dict:
        """Full data processing pipeline."""
        dataset = self.load_dataset()
        formatted_dataset = self.prepare_training_data(dataset)
        tokenized_dataset = self.tokenize_data(formatted_dataset)
        return tokenized_dataset

if __name__ == "__main__":
    # Example usage
    processor = BillSumDataProcessor()
    processed_data = processor.process()
    logger.info("Data processing complete!")
    for split, data in processed_data.items():
        logger.info(f"{split} split size: {len(data)}") 