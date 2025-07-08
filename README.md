# LexiBrief: Legal Document Summarizer 🔍⚖️

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-LexiBrief-blue)](https://huggingface.co/AryanT11/lexibrief-legal-summarizer)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

LexiBrief is an AI-powered legal document summarizer that creates concise, accurate summaries of legal documents. Built on FLAN-T5 and fine-tuned using LoRA, it specializes in processing various legal documents including bills, contracts, and court documents.

## 🌟 Features

- **Specialized Legal Processing**: Trained on BillSum and LexGlue datasets
- **Efficient & Accurate**: Uses LoRA fine-tuning for optimal performance
- **Easy Integration**: Simple API for quick implementation
- **Production Ready**: Optimized for both GPU and CPU environments

## 📁 Project Structure

```
LexiBrief/
├── configs/               # Configuration files
├── outputs/              # Training outputs
│   ├── models/          # Saved model checkpoints
│   └── logs/           # Training logs
├── LexiBrief_Colab.ipynb # Main training notebook
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AryanT11/LexiBrief.git
cd LexiBrief

# Install dependencies
pip install -r requirements.txt
```

### Using the Model

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
model_name = "AryanT11/lexibrief-legal-summarizer"
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

## 📊 Model Details

- **Base Model**: FLAN-T5-base
- **Training Method**: LoRA fine-tuning
- **Datasets**: BillSum and LexGlue
- **Performance Metrics**:
  - ROUGE-1: 0.45
  - ROUGE-2: 0.28
  - ROUGE-L: 0.42

### Training Configuration

```yaml
LoRA Parameters:
  r: 32
  alpha: 32
  target_modules: ["q", "k", "v", "o"]

Training Hyperparameters:
  batch_size: [12, 24]  # train, eval
  learning_rate: 8e-4
  epochs: 2
  max_length: 384
```

## 🔍 Example

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

## ⚠️ Limitations

- Optimized for US legal documents and English language
- Maximum input length: 384 tokens
- Maximum summary length: 128 tokens
- Not a replacement for legal professionals
- May not capture highly technical legal nuances

## 🛠️ Development

To train or modify the model:

1. Open `LexiBrief_Colab.ipynb` in Google Colab
2. Set up your Hugging Face credentials
3. Modify training parameters if needed
4. Run all cells in sequence

## 📫 Contact

- Email: aryan100282@gmail.com
- Hugging Face: [@AryanT11](https://huggingface.co/AryanT11)
- Issues: [GitHub Issues](https://github.com/AryanT11/LexiBrief/issues)

## 📚 Citation

If you use this model in your research, please cite:

```bibtex
@misc{lexibrief2025,
  title={LexiBrief: Legal Document Summarizer},
  author={Aryan Tapkire},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/AryanT11/lexibrief-legal-summarizer}
}
``` 
