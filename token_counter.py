import os
import json
from typing import Dict, List, Optional
from datetime import datetime

class TokenCounter:
    """
    A class to track token usage per user and per model.
    """
    
    def __init__(self, token_data_file: str = "data/token_usage.json"):
        """
        Initialize the TokenCounter with a file to store token usage data.
        
        Args:
            token_data_file (str): Path to the file where token usage data will be stored
        """
        self.token_data_file = token_data_file
        self._ensure_data_directory()
        self.token_data = self._load_token_data()
    
    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        os.makedirs(os.path.dirname(self.token_data_file), exist_ok=True)
    
    def _load_token_data(self) -> Dict:
        """
        Load token usage data from file.
        
        Returns:
            Dict: Token usage data
        """
        if os.path.exists(self.token_data_file):
            try:
                with open(self.token_data_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"users": {}}
        return {"users": {}}
    
    def _save_token_data(self):
        """Save token usage data to file."""
        with open(self.token_data_file, 'w') as f:
            json.dump(self.token_data, f, indent=2)
    
    def add_tokens(self, user_email: str, model_name: str, input_tokens: int, output_tokens: int):
        """
        Add token usage for a user and model.
        
        Args:
            user_email (str): Email of the user
            model_name (str): Name of the model used
            input_tokens (int): Number of input tokens used
            output_tokens (int): Number of output tokens used
        """
        # Initialize user data if not exists
        if user_email not in self.token_data["users"]:
            self.token_data["users"][user_email] = {"models": {}, "total_input_tokens": 0, "total_output_tokens": 0}
        
        # Initialize model data if not exists
        if model_name not in self.token_data["users"][user_email]["models"]:
            self.token_data["users"][user_email]["models"][model_name] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "usage_history": []
            }
        
        # Add token usage
        self.token_data["users"][user_email]["models"][model_name]["input_tokens"] += input_tokens
        self.token_data["users"][user_email]["models"][model_name]["output_tokens"] += output_tokens
        
        # Update user totals
        self.token_data["users"][user_email]["total_input_tokens"] += input_tokens
        self.token_data["users"][user_email]["total_output_tokens"] += output_tokens
        
        # Add usage history entry
        usage_entry = {
            "timestamp": datetime.now().isoformat(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
        self.token_data["users"][user_email]["models"][model_name]["usage_history"].append(usage_entry)
        
        # Save data
        self._save_token_data()
    
    def get_user_token_usage(self, user_email: str) -> Optional[Dict]:
        """
        Get token usage for a specific user.
        
        Args:
            user_email (str): Email of the user
            
        Returns:
            Optional[Dict]: Token usage data for the user or None if user not found
        """
        return self.token_data["users"].get(user_email)
    
    def get_all_users_token_usage(self) -> Dict:
        """
        Get token usage for all users.
        
        Returns:
            Dict: Token usage data for all users
        """
        return self.token_data["users"]
    
    def get_model_token_usage(self, model_name: str) -> Dict:
        """
        Get token usage for a specific model across all users.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dict: Token usage data for the model
        """
        model_usage = {"input_tokens": 0, "output_tokens": 0, "users": {}}
        
        for user_email, user_data in self.token_data["users"].items():
            if model_name in user_data["models"]:
                model_data = user_data["models"][model_name]
                model_usage["input_tokens"] += model_data["input_tokens"]
                model_usage["output_tokens"] += model_data["output_tokens"]
                model_usage["users"][user_email] = {
                    "input_tokens": model_data["input_tokens"],
                    "output_tokens": model_data["output_tokens"]
                }
        
        return model_usage
    
    def get_total_token_usage(self) -> Dict:
        """
        Get total token usage across all users and models.
        
        Returns:
            Dict: Total token usage data
        """
        total_usage = {"input_tokens": 0, "output_tokens": 0}
        
        for user_data in self.token_data["users"].values():
            total_usage["input_tokens"] += user_data["total_input_tokens"]
            total_usage["output_tokens"] += user_data["total_output_tokens"]
        
        return total_usage