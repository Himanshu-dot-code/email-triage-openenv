import os
import json
from openai import OpenAI
from env.environment import EmailTriageEnv


# SAFE environment variable handling (validator-compatible)
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")

# MODEL_NAME may NOT always be injected → must fallback
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")


# Initialize client only if proxy exists
client = None
if API_BASE_URL and API_KEY:
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
    except Exception:
        client = None


TASKS = ["task_easy", "task_medium", "task_hard"]


def fallback_action():
    return {
        "category": "update",
        "priority": "medium",
        "spam": False,
    }


def llm_agent(obs):

    # If proxy unavailable, fallback immediately
    if client is None:
        return fallback_action()

    prompt = f"""
Classify this email.

Return JSON with:
category (meeting/work/update/spam)
priority (high/medium/low)
spam (true/false)

Email:
Subject: {obs.subject}
Body: {obs.body}
Sender: {obs.sender_type}
"""

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        text = response.choices[0].message.content

        action = json.loads(text)

        if not isinstance(action, dict):
            return fallback_action()

        return action

    except Exception:
        return fallback_action()


def run_task(task_name):

    env = EmailTriageEnv()

    obs = env.reset()

    print(f"[START] task={task_name} env=email-triage model={MODEL_NAME}")

    rewards = []
    step = 0

    while True:

        step += 1

        try:
            action_dict = llm_agent(obs)
        except Exception:
            action_dict = fallback_action()

        action = type("Action", (), action_dict)

        try:
            next_obs, reward, done, _ = env.step(action)
        except Exception:
            break

        rewards.append(reward)

        print(
            f"[STEP] step={step} action={action_dict} reward={reward:.2f} done={str(done).lower()} error=null"
        )

        if done:
            break

        obs = next_obs

    success = sum(rewards) > 0

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}"
    )


def main():

    for task in TASKS:
        try:
            run_task(task)
        except Exception:
            print(f"[END] success=false steps=0 rewards=")


if __name__ == "__main__":
    main()
