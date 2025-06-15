---
language: en
tags:
- legal-summarization
- mistral-7b
- fine-tuned
- document-summarization
license: apache-2.0
datasets:
- billsum
-LexGlue
---

# LexiBrief: Legal Document Summarizer

## Model Description

LexiBrief is a fine-tuned version of Mistral-7B-Instruct-v0.1 specifically optimized for legal document summarization. The model has been trained to generate concise, accurate summaries of legal documents while maintaining the essential legal context and key points.

### Training Procedure

The model was fine-tuned using the following approach:

- **Base Model**: mistralai/Mistral-7B-Instruct-v0.1
- **Training Data**: BillSum dataset (US Congressional bills and their summaries)
- **Training Method**: LoRA (Low-Rank Adaptation) fine-tuning
- **Optimization**: 
  - Learning rate: 2e-4
  - Epochs: 3
  - Batch size: Adaptive based on hardware
  - Gradient accumulation steps: 4
  - Warmup ratio: 0.03

### LoRA Configuration
- r: 64
- lora_alpha: 16
- Target modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
- Task type: CAUSAL_LM

## Intended Use

This model is designed for:
- Summarizing legal documents, contracts, and legislative texts
- Generating concise summaries while preserving key legal points
- Assisting legal professionals in document review and analysis

## Limitations

- The model is trained primarily on US legislative documents and may have limited performance on other types of legal documents
- Should not be used as a replacement for legal professionals
- May occasionally generate summaries that miss critical legal details
- Should be used as an assistive tool rather than for making legal decisions

## Bias and Risks

- The model may inherit biases present in the training data
- Legal interpretation requires human expertise and context
- Users should verify generated summaries against original documents
- Not intended for making legal decisions without human review

## Training Data

The model was fine-tuned on the BillSum dataset, which contains:
- US Congressional bills and their corresponding summaries
- A diverse range of legislative topics and writing styles
- Professional summaries created by legislative experts

## Evaluation Results

[Note: Please add your specific evaluation metrics and results after training]

## Example Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "your-username/lexibrief"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare input
text = """[Your legal document here]"""
prompt = f"Instruction: Summarize the following legal document concisely.\n\nDocument: {text}\n\nSummary:"

# Generate summary
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{lexibrief2024,
  author = {[Your Name]},
  title = {LexiBrief: Fine-tuned Legal Document Summarizer},
  year = {2024},
  publisher = {Hugging Face},
  journal = {Hugging Face Model Hub},
  howpublished = {\url{https://huggingface.co/your-username/lexibrief}}
}
```

## Contact

[Your contact information or GitHub profile] 