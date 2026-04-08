from fastapi import FastAPI
from env import SupportEnv

app = FastAPI()

@app.get("/")
def home():
    env = SupportEnv()

    results = []

    for task in env.tasks:
        env.current_task = task

        action = {
            "category": task.get("expected_category"),
            "priority": task.get("expected_priority", "Low"),
            "resolution": task.get("expected_resolution", "guide"),
            "response": "We are sorry, we will resolve your issue"
        }

        _, reward, _, info = env.step(action)

        results.append({
        "task_id": task["id"],
        "difficulty": task["id"].split("_")[0],
        "reward": reward,
        "status": "evaluated",
        "evaluation": {
        "score": reward,
        "explanation": info["reason"]
        }
        })

    # 🔥 NEW: overall average
    avg_reward = round(sum(r["reward"] for r in results) / len(results), 2)

    # 🔥 NEW: difficulty breakdown
    difficulty_groups = {
        "easy": [],
        "medium": [],
        "hard": []
    }

    for r in results:
        difficulty_groups[r["difficulty"]].append(r["reward"])

    difficulty_avg = {
        k: round(sum(v)/len(v), 2) if v else 0
        for k, v in difficulty_groups.items()
    }

    return {
    "status": "running",
    "environment": "support_env",
    "summary": {
        "total_tasks": len(env.tasks),
        "average_reward": avg_reward,
        "difficulty_average": difficulty_avg
    },
    "results": results
    }