from fastapi import FastAPI
from env.environment import EmailTriageEnv

app = FastAPI()

env = EmailTriageEnv()


@app.api_route("/reset", methods=["GET", "POST"])
def reset():

    observation = env.reset()

    return {
        "subject": observation.subject,
        "body": observation.body,
        "sender_type": observation.sender_type
    }