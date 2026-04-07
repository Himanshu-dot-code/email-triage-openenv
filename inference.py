import os
from openai import OpenAI

from env.environment import EmailTriageEnv


# REQUIRED: use hackathon proxy variables
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)

MODEL_NAME = os.getenv("MODEL_NAME")


TASKS = ["task_easy", "task_medium", "task_hard"]


def llm_agent(obs):

    prompt = f"""
You are an enterprise email assistant.

Classify the email into:
category (meeting/work/update/spam)
priority (high/medium/low)
spam (true/false)

Email:
Subject: {obs.subject}
Body: {obs.body}
Sender type: {obs.sender_type}

Return JSON only.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return structured JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    text = response.choices[0].message.content

    try:
        import json
        action = json.loads(text)
    except Exception:
        action = {
            "category": "update",
            "priority": "medium",
            "spam": False
        }

    return action


def run_task(task_name):

    env = EmailTriageEnv()

    obs = env.reset()

    print(f"[START] task={task_name} env=email-triage model={MODEL_NAME}")

    rewards = []
    step = 0

    while True:

        step += 1

        action_dict = llm_agent(obs)

        action = type("Action", (), action_dict)

        next_obs, reward, done, _ = env.step(action)

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
        run_task(task)


if __name__ == "__main__":
    main()