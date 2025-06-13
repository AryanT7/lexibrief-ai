# LexiBrief: AI-Powered Legal Document Summarizer

LexiBrief is an advanced legal document summarization system built on the Mistral-7B-Instruct model, fine-tuned specifically for summarizing U.S. legislative bills. This project demonstrates the application of state-of-the-art language models to the legal domain, utilizing efficient training techniques like QLoRA for resource-optimized fine-tuning.

## Features

- Fine-tuned Mistral-7B-Instruct model using QLoRA (4-bit quantization + LoRA adapters)
- Training on the billsum dataset for specialized legal document summarization
- Evaluation using ROUGE-L and BLEU metrics
- Dual inference support:
  - Local inference with the fine-tuned model
  - Cloud inference via OpenRouter API
- Interactive UI demo using Gradio
- Batch processing capabilities for multiple documents

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LexiBrief.git
cd LexiBrief
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure  

```
LexiBrief/
├── data/                  # Data processing and formatting
├── models/               # Model training and fine-tuning
├── evaluation/           # Evaluation metrics and testing
├── inference/            # Inference API and utilities
├── ui/                   # Gradio UI implementation
├── scripts/              # Utility scripts
└── configs/              # Configuration files
```

## Usage

### Training

To fine-tune the model:
```bash
python scripts/train.py --config configs/training_config.yaml
```

### Evaluation

To evaluate the model:
```bash
python scripts/evaluate.py --model_path path/to/model --test_data path/to/test_data
```

### UI Demo

To launch the Gradio interface:
```bash
python ui/app.py
```

### Batch Processing

To process multiple documents:
```bash
python scripts/batch_process.py --input_dir path/to/input --output_dir path/to/output
```

## Model Performance

The fine-tuned model achieves the following metrics on the test set:
- ROUGE-L: (to be updated after training)
- BLEU: (to be updated after training)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HuggingFace team for the transformers library
- Mistral AI for the base model
- The creators of the billsum dataset

## Citation

If you use this project in your research, please cite:

```bibtex
@software{lexibrief2024,
  author = {Your Name},
  title = {LexiBrief: AI-Powered Legal Document Summarizer},
  year = {2024},
  url = {https://github.com/yourusername/LexiBrief}
}
``` 