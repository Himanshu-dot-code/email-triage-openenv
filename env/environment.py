import json
from env.models import Observation


class EmailTriageEnv:

    def __init__(self):

        with open("data/emails.json") as f:
            self.emails = json.load(f)

        self.index = 0
        self.total_score = 0
        self.last_action = None

    def reset(self):

        self.index = 0
        self.total_score = 0
        self.last_action = None

        email = self.emails[self.index]

        return Observation(
            subject=email["subject"],
            body=email["body"],
            sender_type=email["sender_type"]
        )

    def state(self):

        return {
            "index": self.index,
            "total_score": self.total_score
        }

    def step(self, action):

        email = self.emails[self.index]

        reward = 0.0

        if action.category == email["category"]:
            reward += 0.4

        if action.priority == email["priority"]:
            reward += 0.3

        if action.spam == email["spam"]:
            reward += 0.3

        # exploit-resistance penalty

        if self.last_action:

            if (
                action.category == self.last_action["category"]
                and action.priority == self.last_action["priority"]
                and action.spam == self.last_action["spam"]
            ):
                reward *= 0.85

        self.last_action = {
            "category": action.category,
            "priority": action.priority,
            "spam": action.spam
        }

        self.total_score += reward

        self.index += 1

        done = self.index >= len(self.emails)

        if not done:

            next_email = self.emails[self.index]

            observation = Observation(
                subject=next_email["subject"],
                body=next_email["body"],
                sender_type=next_email["sender_type"]
            )

        else:

            observation = None

        return observation, reward, done, {}