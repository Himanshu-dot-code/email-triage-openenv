from env.tasks import TASKS


class EmailTriageGrader:

    def __init__(self, task_name):

        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}")

        self.task = TASKS[task_name]

        # realistic enterprise importance weighting
        self.field_weights = {
            "category": 0.4,
            "priority": 0.35,
            "spam": 0.25
        }

    def grade(self, prediction, ground_truth):

        required_fields = self.task["fields_required"]

        total_weight = 0
        earned_weight = 0

        for field in required_fields:

            weight = self.field_weights[field]
            total_weight += weight

            if prediction.get(field) == ground_truth.get(field):
                earned_weight += weight

        return earned_weight / total_weight