import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
from typing import Dict, Optional
import yaml
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.openrouter_inference import summarize_bill_with_mistral

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LexiBriefUI:
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """
        Initialize the UI application.
        
        Args:
            config_path: Path to the configuration file
        """
        self.load_config(config_path)
        self.setup_model()
        
    def load_config(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def setup_model(self):
        """Initialize the model and tokenizer."""
        logger.info("Setting up model...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        # We'll load the model on first use to make the UI start faster
        
    def load_model_if_needed(self):
        """Load the model and tokenizer if not already loaded."""
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model"]["name"],
                device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model"]["name"],
                trust_remote_code=True
            )
    
    def generate_summary_local(self, text: str, max_length: int = 2048) -> str:
        """Generate summary using the local model."""
        self.load_model_if_needed()
        
        prompt = f"<s>[INST] Summarize the following legal bill:\n\n{text} [/INST]"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
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
    
    def process_text(
        self,
        text: str,
        use_api: bool = False,
        max_length: int = 2048
    ) -> Dict[str, str]:
        """Process the input text and generate summary."""
        try:
            if not text.strip():
                return {
                    "summary": "Please enter some text to summarize.",
                    "error": None
                }
            
            if use_api:
                summary = summarize_bill_with_mistral(text)
            else:
                summary = self.generate_summary_local(text, max_length)
            
            return {
                "summary": summary,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {
                "summary": None,
                "error": f"Error generating summary: {str(e)}"
            }
    
    def create_ui(self) -> gr.Interface:
        """Create the Gradio interface."""
        theme = gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
            neutral_hue="slate",
            text_size=gr.themes.sizes.text_lg,
        ).set(
            body_background_fill="*neutral_950",
            body_background_fill_dark="*neutral_950",
            body_text_color="*neutral_100",
            body_text_color_dark="*neutral_100",
            block_background_fill="*neutral_900",
            block_background_fill_dark="*neutral_900",
            block_label_text_color="*neutral_200",
            block_label_text_color_dark="*neutral_200",
            input_background_fill="*neutral_800",
            input_background_fill_dark="*neutral_800",
            input_border_color="*neutral_700",
            input_border_color_dark="*neutral_700",
            button_primary_background_fill="*primary_600",
            button_primary_background_fill_dark="*primary_600",
            button_primary_text_color="white",
            button_primary_text_color_dark="white",
            checkbox_background_color="*neutral_800",
            checkbox_background_color_dark="*neutral_800",
            checkbox_border_color="*neutral_600",
            checkbox_border_color_dark="*neutral_600",
            slider_color="*primary_500",
            slider_color_dark="*primary_500",
        )

        with gr.Blocks(title="LexiBrief - Legal Document Summarizer", theme=theme) as interface:
            gr.Markdown("""
            # LexiBrief - Legal Document Summarizer
            
            Upload or paste a legal document (e.g., a bill) and get a concise, human-readable summary.
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    input_text = gr.Textbox(
                        label="Input Legal Document",
                        placeholder="Paste your legal document here...",
                        lines=10
                    )
                    
                with gr.Column(scale=2):
                    output_text = gr.Textbox(
                        label="Generated Summary",
                        lines=10,
                        interactive=False
                    )
                    error_text = gr.Textbox(
                        label="Error (if any)",
                        visible=False,
                        interactive=False
                    )
            
            with gr.Row():
                with gr.Column():
                    use_api = gr.Checkbox(
                        label="Use OpenRouter API (requires API key in .env file)",
                        value=False
                    )
                    max_length = gr.Slider(
                        label="Max Input Length",
                        minimum=512,
                        maximum=4096,
                        value=2048,
                        step=128
                    )
                
                with gr.Column():
                    submit_btn = gr.Button("Generate Summary", variant="primary")
            
            # Example inputs
            gr.Examples(
                examples=[
                    ["This bill would establish a new program to provide grants to states for..."],
                    ["The purpose of this act is to amend the existing regulations regarding..."]
                ],
                inputs=input_text,
                label="Example Documents"
            )
            
            # Handle submit
            submit_btn.click(
                fn=lambda x, y, z: (
                    self.process_text(x, y, z)["summary"],
                    self.process_text(x, y, z)["error"]
                ),
                inputs=[input_text, use_api, max_length],
                outputs=[output_text, error_text]
            ).then(
                fn=lambda x: gr.update(visible=bool(x)),
                inputs=[error_text],
                outputs=[error_text]
            )
            
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        interface = self.create_ui()
        interface.launch(**kwargs)

if __name__ == "__main__":
    # Create and launch the UI
    app = LexiBriefUI()
    app.launch(share=True) 