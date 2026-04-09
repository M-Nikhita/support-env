import json
import random
from typing import Dict, Any
from graders import grade

class SupportEnv:
    def __init__(self):
        with open("tasks.json") as f:
            self.tasks = json.load(f)
        self.current_task = None
        self.history = []

    def reset(self) -> Dict[str, Any]:
        self.current_task = random.choice(self.tasks)
        self.history = []
        return {
            "ticket": self.current_task["ticket"]
        }

    def state(self) -> Dict[str, Any]:
        return self.current_task

    def step(self, action: Dict[str, Any]):
        # store action history
        self.history.append(action)

        reward = grade(self.current_task, action)

        # penalty for repeating same action
        if len(self.history) > 1 and self.history[-1] == self.history[-2]:
            reward -= 0.1

        # bonus for correct resolution
        if action.get("resolution") == self.current_task.get("expected_resolution"):
            reward += 0.05

        reward = round(reward, 2)
        reward = max(0.01, min(0.99, reward))

        info = {
            "reason": "reward based on category, priority, resolution, and response quality"
        }

        return (
            {
                "ticket": self.current_task["ticket"],
                "history": self.history
            },
            reward,
            True,
            info
        )
