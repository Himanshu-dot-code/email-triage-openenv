from enum import Enum


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


TASKS = {

    "task_easy": {

        "difficulty": TaskDifficulty.EASY,

        "description": (
            "Classify email category only. "
            "Represents baseline enterprise inbox classification where the agent "
            "routes messages into workflow buckets."
        ),

        "fields_required": ["category"],

        "evaluation_focus": "workflow routing"
    },

    "task_medium": {

        "difficulty": TaskDifficulty.MEDIUM,

        "description": (
            "Classify email category and priority. "
            "Represents assistant-level decision-making where the agent must "
            "support scheduling and urgency handling."
        ),

        "fields_required": ["category", "priority"],

        "evaluation_focus": "routing + prioritization"
    },

    "task_hard": {

        "difficulty": TaskDifficulty.HARD,

        "description": (
            "Classify category, priority and spam detection simultaneously. "
            "Represents full enterprise inbox automation pipeline combining "
            "classification, urgency estimation and filtering."
        ),

        "fields_required": ["category", "priority", "spam"],

        "evaluation_focus": "routing + prioritization + filtering"
    }
}