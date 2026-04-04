from fastapi import FastAPI
from env.environment import EmailTriageEnv

app = FastAPI()

env = EmailTriageEnv()


@app.get("/")
def root():
    return {"status": "Email Triage OpenEnv running"}


@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(action: dict):
    action_obj = type("Action", (), action)
    obs, reward, done, info = env.step(action_obj)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }