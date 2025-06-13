import torch
import numpy as np
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_metric
import logging
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download required NLTK data
nltk.download('punkt', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LexiBriefEvaluator:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 2048
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to run inference on
            max_length: Maximum sequence length for generation
        """
        self.device = device
        self.max_length = max_length
        
        logger.info(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
        
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
        # Extract the generated summary from the response
        summary = summary.split("[/INST]")[-1].strip()
        return summary
    
    def calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        scores = self.rouge_scorer.score(reference, generated)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure
        }
    
    def calculate_bleu_score(self, generated: str, reference: str) -> float:
        """Calculate BLEU score."""
        generated_tokens = nltk.word_tokenize(generated.lower())
        reference_tokens = [nltk.word_tokenize(reference.lower())]
        return sentence_bleu(reference_tokens, generated_tokens, smoothing_function=self.smoothing)
    
    def evaluate_sample(self, text: str, reference_summary: str) -> Dict[str, float]:
        """Evaluate a single sample."""
        generated_summary = self.generate_summary(text)
        
        # Calculate metrics
        rouge_scores = self.calculate_rouge_scores(generated_summary, reference_summary)
        bleu_score = self.calculate_bleu_score(generated_summary, reference_summary)
        
        metrics = {
            **rouge_scores,
            "bleu": bleu_score
        }
        
        return {
            "generated_summary": generated_summary,
            "metrics": metrics
        }
    
    def evaluate_dataset(self, dataset, num_samples: int = None) -> Dict:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset: Dataset to evaluate on
            num_samples: Number of samples to evaluate (None for all)
        """
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        all_metrics = []
        generated_summaries = []
        
        for example in tqdm(dataset, desc="Evaluating"):
            result = self.evaluate_sample(example["text"], example["summary"])
            all_metrics.append(result["metrics"])
            generated_summaries.append(result["generated_summary"])
            
        # Calculate average metrics
        avg_metrics = {
            metric: np.mean([m[metric] for m in all_metrics])
            for metric in all_metrics[0].keys()
        }
        
        return {
            "average_metrics": avg_metrics,
            "all_metrics": all_metrics,
            "generated_summaries": generated_summaries
        }
    
    def save_evaluation_results(self, results: Dict, output_path: str):
        """Save evaluation results to a file."""
        import json
        
        logger.info(f"Saving evaluation results to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    # Example usage
    from datasets import load_dataset
    
    # Load test dataset
    dataset = load_dataset("billsum")["test"]
    
    # Initialize evaluator
    evaluator = LexiBriefEvaluator("path/to/your/fine-tuned/model")
    
    # Run evaluation
    results = evaluator.evaluate_dataset(dataset, num_samples=10)
    
    # Print results
    print("\nEvaluation Results:")
    for metric, value in results["average_metrics"].items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    evaluator.save_evaluation_results(results, "evaluation_results.json") 