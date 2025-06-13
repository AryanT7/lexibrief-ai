import argparse
import os
import json
import logging
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 2048,
        batch_size: int = 1
    ):
        """
        Initialize the batch processor.
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to run inference on
            max_length: Maximum sequence length for generation
            batch_size: Batch size for processing
        """
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        
        logger.info(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    def read_document(self, file_path: str) -> str:
        """Read a document from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def generate_summary(self, text: str) -> str:
        """Generate a summary for the input text."""
        prompt = f"<s>[INST] Summarize the following legal bill:\n\n{text} [/INST]"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = summary.split("[/INST]")[-1].strip()
        return summary
    
    def process_batch(self, file_paths: List[str]) -> List[Dict]:
        """Process a batch of documents."""
        results = []
        
        for file_path in tqdm(file_paths, desc="Processing documents"):
            try:
                text = self.read_document(file_path)
                summary = self.generate_summary(text)
                
                results.append({
                    "file_path": file_path,
                    "summary": summary,
                    "status": "success",
                    "error": None
                })
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                results.append({
                    "file_path": file_path,
                    "summary": None,
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    def save_results(self, results: List[Dict], output_dir: str):
        """Save processing results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed JSON results
        json_path = os.path.join(output_dir, "batch_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV summary
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "batch_results.csv")
        df.to_csv(csv_path, index=False)
        
        # Save individual summaries
        summaries_dir = os.path.join(output_dir, "summaries")
        os.makedirs(summaries_dir, exist_ok=True)
        
        for result in results:
            if result["status"] == "success":
                file_name = Path(result["file_path"]).stem
                summary_path = os.path.join(summaries_dir, f"{file_name}_summary.txt")
                with open(summary_path, 'w') as f:
                    f.write(result["summary"])
        
        logger.info(f"Results saved to {output_dir}")
        logger.info(f"Processed {len(results)} documents")
        logger.info(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
        logger.info(f"Failed: {sum(1 for r in results if r['status'] == 'error')}")

def main():
    parser = argparse.ArgumentParser(description="Batch process legal documents for summarization")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input documents")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum input length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--file_pattern", type=str, default="*.txt", help="File pattern to match (e.g., *.txt, *.pdf)")
    
    args = parser.parse_args()
    
    # Get list of files to process
    input_files = list(Path(args.input_dir).glob(args.file_pattern))
    if not input_files:
        logger.error(f"No files found matching pattern {args.file_pattern} in {args.input_dir}")
        return
    
    logger.info(f"Found {len(input_files)} files to process")
    
    # Initialize processor
    processor = BatchProcessor(
        model_path=args.model_path,
        max_length=args.max_length,
        batch_size=args.batch_size
    )
    
    # Process files
    results = processor.process_batch([str(f) for f in input_files])
    
    # Save results
    processor.save_results(results, args.output_dir)

if __name__ == "__main__":
    main() 