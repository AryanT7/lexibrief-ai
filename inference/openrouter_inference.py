from dotenv import load_dotenv
import os
import requests
import logging
from typing import Optional

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenRouterInference:
    def __init__(self):
        """Initialize the OpenRouter inference client."""
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Please ensure you have set the OPENROUTER_API_KEY "
                "environment variable in your .env file."
            )
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "mistralai/mistral-7b-instruct-v0.2"  # Updated model name
    
    def _get_headers(self) -> dict:
        """Get the headers for the API request."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/your-username/LexiBrief",  # Replace with your repo
            "X-Title": "LexiBrief",
            "Content-Type": "application/json"
        }
    
    def summarize_bill(self, text: str) -> str:
        """
        Summarize a legal bill using the OpenRouter API.
        
        Args:
            text: The text of the legal bill to summarize
            
        Returns:
            str: The generated summary
            
        Raises:
            Exception: If the API request fails
        """
        try:
            # Log the API key existence (not the key itself)
            logger.info(f"API Key present: {bool(self.api_key)}")
            
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a legal expert that specializes in summarizing legal documents clearly and concisely."
                    },
                    {
                        "role": "user",
                        "content": f"Please provide a clear and concise summary of the following legal bill, highlighting its key points and main objectives:\n\n{text}"
                    }
                ]
            }
            
            logger.info(f"Making request to OpenRouter API with model: {self.model}")
            
            response = requests.post(
                self.base_url,
                headers=self._get_headers(),
                json=data,
                timeout=30  # Added timeout
            )
            
            logger.info(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Successfully received response from OpenRouter")
                return result["choices"][0]["message"]["content"]
            else:
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except requests.exceptions.Timeout:
            error_msg = "Request to OpenRouter API timed out"
            logger.error(error_msg)
            raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error when calling OpenRouter API: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Error in OpenRouter inference: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

# Create a singleton instance
openrouter = OpenRouterInference()

def summarize_bill_with_mistral(text: str) -> str:
    """
    Public function to summarize a legal bill using Mistral through OpenRouter.
    
    Args:
        text: The text of the legal bill to summarize
        
    Returns:
        str: The generated summary
    """
    return openrouter.summarize_bill(text) 