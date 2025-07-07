import os
import json
from typing import Dict, List, Optional
from datetime import datetime


class AnalysisHistory:
    """
    A class to track analysis history per user.
    """

    def __init__(self, history_data_file: str = "data/analysis_history.json"):
        """
        Initialize the AnalysisHistory with a file to store analysis history data.

        Args:
            history_data_file (str): Path to the file where analysis history data will be stored
        """
        self.history_data_file = history_data_file
        self._ensure_data_directory()
        self.history_data = self._load_history_data()

    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        os.makedirs(os.path.dirname(self.history_data_file), exist_ok=True)

    def _load_history_data(self) -> Dict:
        """
        Load analysis history data from file.

        Returns:
            Dict: Analysis history data
        """
        if os.path.exists(self.history_data_file):
            try:
                with open(self.history_data_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"users": {}}
        return {"users": {}}

    def _save_history_data(self):
        """Save analysis history data to file."""
        with open(self.history_data_file, 'w') as f:
            json.dump(self.history_data, f, indent=2, ensure_ascii=False)

    def add_analysis(self, user_email: str, analysis_data: Dict):
        """
        Add an analysis to user's history.

        Args:
            user_email (str): Email of the user
            analysis_data (Dict): The analysis data to store
        """
        # Initialize user data if not exists
        if user_email not in self.history_data["users"]:
            self.history_data["users"][user_email] = {
                "analyses": [],
                "total_analyses": 0
            }

        # Add timestamp and unique ID to analysis
        analysis_entry = {
            "id": f"{user_email}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.history_data['users'][user_email]['analyses'])}",
            "timestamp": datetime.now().isoformat(),
            "requirement": analysis_data.get('requirement', ''),
            "result": analysis_data.get('result', {}),
            "model_used": analysis_data.get('model_used', 'unknown'),
            "summary": {
                "implementation_type": analysis_data.get('result', {}).get('response',
                                                                           {}).get(
                    'implementation_type', 'unknown'),
                "confidence_score": analysis_data.get('result', {}).get('response',
                                                                        {}).get(
                    'analysis_result', {}).get('confidence_score', 0),
                "total_tokens": analysis_data.get('result', {}).get('token_usage',
                                                                    {}).get(
                    'total_tokens', 0)
            }
        }

        # Add to user's analyses
        self.history_data["users"][user_email]["analyses"].append(analysis_entry)
        self.history_data["users"][user_email]["total_analyses"] += 1

        # Keep only last 50 analyses per user to prevent file from growing too large
        if len(self.history_data["users"][user_email]["analyses"]) > 50:
            self.history_data["users"][user_email]["analyses"] = \
            self.history_data["users"][user_email]["analyses"][-50:]

        # Save data
        self._save_history_data()

        return analysis_entry["id"]

    def get_user_history(self, user_email: str, limit: Optional[int] = None) -> List[
        Dict]:
        """
        Get analysis history for a specific user.

        Args:
            user_email (str): Email of the user
            limit (Optional[int]): Maximum number of analyses to return (most recent first)

        Returns:
            List[Dict]: List of analysis entries for the user
        """
        if user_email not in self.history_data["users"]:
            return []

        analyses = self.history_data["users"][user_email]["analyses"]

        # Sort by timestamp descending (most recent first)
        sorted_analyses = sorted(analyses, key=lambda x: x["timestamp"], reverse=True)

        if limit:
            return sorted_analyses[:limit]

        return sorted_analyses

    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict]:
        """
        Get a specific analysis by its ID.

        Args:
            analysis_id (str): The analysis ID

        Returns:
            Optional[Dict]: The analysis data or None if not found
        """
        # Extract user email from analysis ID
        try:
            user_email = analysis_id.split('_')[0] + '@' + \
                         analysis_id.split('@')[1].split('_')[0]
        except:
            return None

        user_analyses = self.get_user_history(user_email)

        for analysis in user_analyses:
            if analysis["id"] == analysis_id:
                return analysis

        return None

    def delete_analysis(self, user_email: str, analysis_id: str) -> bool:
        """
        Delete a specific analysis from user's history.

        Args:
            user_email (str): Email of the user
            analysis_id (str): The analysis ID to delete

        Returns:
            bool: True if analysis was deleted, False if not found
        """
        if user_email not in self.history_data["users"]:
            return False

        analyses = self.history_data["users"][user_email]["analyses"]

        for i, analysis in enumerate(analyses):
            if analysis["id"] == analysis_id:
                del analyses[i]
                self.history_data["users"][user_email]["total_analyses"] -= 1
                self._save_history_data()
                return True

        return False

    def clear_user_history(self, user_email: str) -> bool:
        """
        Clear all analysis history for a specific user.

        Args:
            user_email (str): Email of the user

        Returns:
            bool: True if history was cleared, False if user not found
        """
        if user_email not in self.history_data["users"]:
            return False

        self.history_data["users"][user_email]["analyses"] = []
        self.history_data["users"][user_email]["total_analyses"] = 0
        self._save_history_data()
        return True

    def get_user_stats(self, user_email: str) -> Dict:
        """
        Get statistics for a user's analysis history.

        Args:
            user_email (str): Email of the user

        Returns:
            Dict: Statistics about the user's analyses
        """
        if user_email not in self.history_data["users"]:
            return {
                "total_analyses": 0,
                "configuration_analyses": 0,
                "development_analyses": 0,
                "total_tokens_used": 0,
                "models_used": {},
                "avg_confidence_score": 0
            }

        analyses = self.history_data["users"][user_email]["analyses"]

        stats = {
            "total_analyses": len(analyses),
            "configuration_analyses": 0,
            "development_analyses": 0,
            "total_tokens_used": 0,
            "models_used": {},
            "confidence_scores": []
        }

        for analysis in analyses:
            summary = analysis.get("summary", {})

            # Count implementation types
            impl_type = summary.get("implementation_type", "").lower()
            if "configuration" in impl_type:
                stats["configuration_analyses"] += 1
            elif "development" in impl_type:
                stats["development_analyses"] += 1

            # Sum tokens
            stats["total_tokens_used"] += summary.get("total_tokens", 0)

            # Count models used
            model = analysis.get("model_used", "unknown")
            stats["models_used"][model] = stats["models_used"].get(model, 0) + 1

            # Collect confidence scores
            confidence = summary.get("confidence_score", 0)
            if confidence > 0:
                stats["confidence_scores"].append(confidence)

        # Calculate average confidence score
        if stats["confidence_scores"]:
            stats["avg_confidence_score"] = sum(stats["confidence_scores"]) / len(
                stats["confidence_scores"])
        else:
            stats["avg_confidence_score"] = 0

        # Remove the list as it's only used for calculation
        del stats["confidence_scores"]

        return stats
