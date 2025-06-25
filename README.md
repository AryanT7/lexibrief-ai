# LexiBrief: Legal Document Summarizer

LexiBrief is a fine-tuned version of Mistral-7B-Instruct specifically optimized for legal document summarization. The model uses LoRA (Low-Rank Adaptation) for efficient fine-tuning and is trained on the BillSum dataset.

## Project Structure

```
LexiBrief/
├── configs/               # Configuration files
├── outputs/              # Training outputs
│   ├── models/          # Saved model checkpoints
│   ├── logs/           # Training logs
│   └── results/        # Evaluation results
├── LexiBrief_Colab.ipynb # Main training notebook
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Setup and Usage

1. Open `LexiBrief_Colab.ipynb` in Google Colab
2. Run the notebook cells in sequence
3. When prompted, log in to your Hugging Face account

The notebook will:
- Set up the environment
- Install dependencies
- Load and preprocess the BillSum dataset
- Fine-tune Mistral-7B-Instruct using LoRA
- Provide a Gradio interface for testing

## Model Details

- Base Model: mistralai/Mistral-7B-Instruct-v0.1
- Training Method: LoRA fine-tuning with fp16 precision
- Dataset: BillSum (US Congressional bills and their summaries)
- Hardware: Optimized for Google Colab's T4 GPU

### Training Configuration

- Learning rate: 2e-4
- Epochs: 3
- Batch size: 4 (GPU) / 1 (CPU)
- Gradient accumulation steps: 4
- Mixed precision: fp16 (GPU) / no (CPU)

### LoRA Configuration

- r: 64
- lora_alpha: 16
- Target modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
- Task type: CAUSAL_LM

## Requirements

See `requirements.txt` for the full list of dependencies. Key requirements:
- transformers==4.40.2
- peft==0.10.0
- torch>=2.0.0
- accelerate==0.26.0

## License

Apache License 2.0

## Contact

For questions or issues, please open a GitHub issue. 