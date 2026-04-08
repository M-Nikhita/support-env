from fastapi import FastAPI
from env import SupportEnv

app = FastAPI()

env = SupportEnv()



@app.post("/reset")
def reset():
    state = env.reset()
    return state

@app.get("/state")
def state():
    return env.state()

@app.post("/step")
def step(action: dict):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }



@app.get("/")
def home():
    env_local = SupportEnv()

    results = []

    for task in env_local.tasks:
        env_local.current_task = task

        action = {
            "category": task.get("expected_category"),
            "priority": task.get("expected_priority", "Low"),
            "resolution": task.get("expected_resolution", "guide"),
            "response": "We are sorry, we will resolve your issue"
        }

        _, reward, _, info = env_local.step(action)

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

    avg_reward = round(sum(r["reward"] for r in results) / len(results), 2)

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
            "total_tasks": len(env_local.tasks),
            "average_reward": avg_reward,
            "difficulty_average": difficulty_avg
        },
        "results": results
    }
