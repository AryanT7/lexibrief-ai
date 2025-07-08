---
language:
- en
tags:
- legal
- summarization
- t5
- flan-t5
- peft
- lora
- legal-nlp
- document-summarization
- billsum
- lexglue
license: apache-2.0
datasets:
- billsum
- lexglue
model-index:
- name: LexiBrief Legal Summarizer
  results:
  - task:
      type: summarization
      name: Legal Document Summarization
    dataset:
      name: billsum
      type: billsum
      split: test
    metrics:
      - name: Training Loss
        type: loss
        value: 1.5808
      - name: ROUGE-1
        type: rouge
        value: 0.45
      - name: ROUGE-2
        type: rouge
        value: 0.28
      - name: ROUGE-L
        type: rouge
        value: 0.42
---

# LexiBrief: Legal Document Summarizer

## Model Description

This model is a fine-tuned version of [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) specifically optimized for legal document summarization. It has been trained on a combination of the BillSum and LexGlue datasets, making it particularly effective at summarizing various types of legal documents including:
- Legislative bills
- Legal contracts
- Court documents
- Legal agreements
- Regulatory documents

The model uses LoRA (Low-Rank Adaptation) for efficient fine-tuning while maintaining the base model's strong language understanding capabilities. This approach allows the model to:
- Maintain the general language understanding from FLAN-T5
- Develop specialized legal domain expertise
- Achieve high-quality summarization with minimal training resources

## Key Features and Benefits

1. **Legal Domain Specialization**:
   - Trained specifically on legal documents
   - Understands legal terminology and context
   - Maintains formal language appropriate for legal documents

2. **Performance Advantages**:
   - Generates concise yet comprehensive summaries
   - Preserves critical legal details
   - Handles complex legal terminology effectively
   - Maintains document structure awareness

3. **Technical Improvements**:
   - Optimized sequence length for legal documents
   - Enhanced attention to legal terms and clauses
   - Efficient processing of long documents
   - Memory-efficient thanks to LoRA adaptation

## Intended Uses & Limitations

### Intended Uses
- Summarizing legislative bills and legal documents
- Creating executive summaries of legal agreements
- Quick document review and analysis
- Legal research assistance
- Contract analysis and summary generation

### Limitations
- The model is primarily trained on US legislative bills and legal documents
- Input documents should be in English
- Maximum input length is 384 tokens
- Generated summaries are limited to 128 tokens
- May not capture extremely technical legal nuances
- Should not be used as a replacement for legal professionals
- Not suitable for non-English legal documents

## Training and Evaluation Data

### Training Data
The model was trained on:
1. **BillSum Dataset**:
   - Contains US Congressional bills
   - Provides high-quality summaries
   - Focuses on legislative language

2. **LexGlue Components**:
   - Legal document corpus
   - Various legal document types
   - Professional-grade annotations

### Training Configuration
- **LoRA Parameters**:
  - Rank (r): 32
  - Alpha: 32
  - Target Modules: q, k, v, o attention layers
  - Task Type: SEQ_2_SEQ_LM

- **Training Hyperparameters**:
  - Batch Size: 12 (train), 24 (eval)
  - Learning Rate: 8e-4
  - Epochs: 2
  - Max Input Length: 384 tokens
  - Max Output Length: 128 tokens
  - Mixed Precision: bfloat16

## Performance and Evaluation

The model demonstrates strong performance in legal document summarization:
- Maintains high factual accuracy
- Preserves critical legal details
- Generates coherent and structured summaries
- Handles complex legal terminology effectively

### Metrics:
- Training Loss: 1.5808
- ROUGE Scores:
  - ROUGE-1: ~0.45
  - ROUGE-2: ~0.28
  - ROUGE-L: ~0.42

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
model_name = "AryanT11/lexibrief-legal-summarizer"  # Replace with actual model ID
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Prepare input
text = "Your legal document here..."
inputs = tokenizer(f"summarize legal document: {text}", 
                  return_tensors="pt", 
                  max_length=384,
                  truncation=True)

# Generate summary
outputs = model.generate(**inputs, 
                        max_length=128,
                        temperature=0.7,
                        do_sample=True)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

## Example Output

Input:
```
SECTION 1. SHORT TITLE.
This Act may be cited as the "Sample Legal Document Act of 2024".

SECTION 2. PURPOSE.
The purpose of this Act is to establish guidelines for legal document processing
and ensure compliance with regulatory requirements.
```

Output:
```
This Act, known as the Sample Legal Document Act of 2024, establishes guidelines
for legal document processing and ensures regulatory compliance.
```

## Citation

If you use this model, please cite:

```bibtex
@misc{lexibrief2025,
  title={LexiBrief: Legal Document Summarizer},
  author={Aryan Tapkire},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/AryanT11/lexibrief-legal-summarizer}
}
```

## Contact

For questions, issues, or feedback about this model, please:
1. Contact me on aryan100282@gmail.com
2. Open an issue on the model repository 