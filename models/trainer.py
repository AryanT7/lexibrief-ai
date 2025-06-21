import os
import yaml
import torch
import logging
from typing import Dict, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from data.data_processor import BillSumDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LexiBriefTrainer:
    def __init__(self, config_path: str):
        """
        Initialize the trainer with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.setup_directories()
        self.setup_model_and_tokenizer()
        self.setup_data()
        
    def setup_directories(self):
        """Create necessary directories for outputs and logs."""
        os.makedirs(self.config["output"]["output_dir"], exist_ok=True)
        os.makedirs(self.config["output"]["logging_dir"], exist_ok=True)
        
    def setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer with QLoRA configuration for GPU."""
        logger.info("Setting up model and tokenizer...")
        
        # Setup quantization config for GPU efficiency
        # compute_dtype = torch.bfloat16 if self.config["hardware"]["mixed_precision"] == "bf16" else torch.float16
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model"]["name"],
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        
        # Prepare model for k-bit training
        # self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA configuration
        peft_config = LoraConfig(
            r=self.config["peft"]["r"],
            lora_alpha=self.config["peft"]["lora_alpha"],
            lora_dropout=self.config["peft"]["lora_dropout"],
            bias=self.config["peft"]["bias"],
            task_type=self.config["peft"]["task_type"],
            target_modules=self.config["peft"]["target_modules"],
            inference_mode=self.config["peft"]["inference_mode"]
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"]["name"],
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def setup_data(self):
        """Prepare the dataset for training."""
        logger.info("Preparing dataset...")
        processor = BillSumDataProcessor(
            model_name=self.config["model"]["name"],
            max_length=self.config["model"]["max_length"],
            train_size=self.config["data"]["train_size"],
            seed=self.config["training"]["seed"]
        )
        self.dataset = processor.process()
        
    def get_training_arguments(self) -> TrainingArguments:
        """Configure training arguments."""
        return TrainingArguments(
            output_dir=self.config["output"]["output_dir"],
            num_train_epochs=self.config["training"]["num_train_epochs"],
            per_device_train_batch_size=self.config["training"]["per_device_train_batch_size"],
            per_device_eval_batch_size=self.config["training"]["per_device_eval_batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            learning_rate=self.config["training"]["learning_rate"],
            warmup_ratio=self.config["training"]["warmup_ratio"],
            weight_decay=self.config["training"]["weight_decay"],
            logging_steps=self.config["training"]["logging_steps"],
            eval_steps=self.config["training"]["eval_steps"],
            save_steps=self.config["training"]["save_steps"],
            max_grad_norm=self.config["training"]["max_grad_norm"],
            save_total_limit=self.config["training"]["save_total_limit"],
            evaluation_strategy=self.config["training"]["evaluation_strategy"],
            save_strategy=self.config["training"]["save_strategy"],
            load_best_model_at_end=self.config["training"]["load_best_model_at_end"],
            metric_for_best_model=self.config["training"]["metric_for_best_model"],
            greater_is_better=self.config["training"]["greater_is_better"],
            report_to=self.config["output"]["report_to"],
            bf16=self.config["hardware"]["mixed_precision"] == "bf16",
            fp16=self.config["hardware"]["mixed_precision"] == "fp16"
        )
        
    def train(self):
        """Execute the training process."""
        logger.info("Starting training...")
        
        training_args = self.get_training_arguments()
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            args=training_args,
            tokenizer=self.tokenizer,
            packing=False
        )
        
        # Train the model
        train_result = trainer.train()
        metrics = train_result.metrics
        
        # Save the final model
        trainer.save_model()
        
        # Log and save metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        logger.info("Training completed successfully!")
        return metrics

if __name__ == "__main__":
    # Example usage
    trainer = LexiBriefTrainer("configs/training_config.yaml")
    metrics = trainer.train() 