from fastapi import FastAPI
from env import SupportEnv
import uvicorn

app = FastAPI()
env = SupportEnv()

@app.post("/reset")
def reset():
    return env.reset()

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

# 🔥 THIS IS REQUIRED
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

# 🔥 ALSO REQUIRED
if __name__ == "__main__":
    main()
