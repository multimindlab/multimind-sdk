from typing import Optional, Dict, Any, List
import os
import logging
import subprocess
from pathlib import Path
from dotenv import load_dotenv

import openai
from anthropic import Anthropic
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelWrapper:
    def __init__(self, env_path: Optional[str] = None):
        # Load environment variables
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()  # Look for .env in current directory
            
        # Load API keys from environment
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.claude_key = os.getenv('CLAUDE_API_KEY')
        self.hf_token = os.getenv('HF_TOKEN')
        
        # Initialize clients
        if self.openai_key:
            openai.api_key = self.openai_key
            logger.info("OpenAI API key loaded")
        if self.claude_key:
            self.claude = Anthropic(api_key=self.claude_key)
            logger.info("Claude API key loaded")
            
    def available_models(self) -> List[str]:
        """Check which models are available based on API keys and local installations."""
        available = []
        
        # Check API-based models
        if self.openai_key:
            available.append("openai")
        if self.claude_key:
            available.append("claude")
        if self.hf_token:
            available.append("huggingface")
            
        # Check Ollama installation
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                available.append("ollama")
                logger.info("Ollama installation detected")
        except FileNotFoundError:
            logger.warning("Ollama not installed")
            
        logger.info(f"Available models: {', '.join(available)}")
        return available
            
    def query_openai(self, prompt: str) -> str:
        logger.info("Querying OpenAI model")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message['content']
        
    def query_claude(self, prompt: str) -> str:
        logger.info("Querying Claude model")
        response = self.claude.messages.create(
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
        
    def query_ollama(self, prompt: str, model: str = "mistral") -> str:
        logger.info(f"Querying Ollama model: {model}")
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True
        )
        return result.stdout
        
    def query_huggingface(self, prompt: str, model_id: str = "mistralai/Mistral-7B-v0.1") -> str:
        logger.info(f"Querying Hugging Face model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_id, token=self.hf_token)
        
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=200)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    def query_model(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Query a specific model with error handling and logging."""
        try:
            # Check if model is available
            if model not in self.available_models():
                raise ValueError(f"Model {model} is not available. Available models: {', '.join(self.available_models())}")
                
            response = ""
            if model == "openai":
                response = self.query_openai(prompt)
            elif model == "claude":
                response = self.query_claude(prompt)
            elif model == "ollama":
                response = self.query_ollama(prompt, kwargs.get('ollama_model', 'mistral'))
            elif model == "huggingface":
                response = self.query_huggingface(prompt, kwargs.get('hf_model_id'))
            else:
                raise ValueError(f"Unsupported model: {model}")
                
            return {
                "status": "success",
                "model": model,
                "response": response
            }
        except Exception as e:
            logger.error(f"Error querying {model}: {str(e)}")
            return {
                "status": "error",
                "model": model,
                "error": str(e)
            } 