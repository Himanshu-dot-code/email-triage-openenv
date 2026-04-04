import os
from env.environment import EmailTriageEnv
from env.grader import EmailTriageGrader


# Required environment variables (hackathon spec)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "heuristic-agent")
HF_TOKEN = os.getenv("HF_TOKEN")


TASKS = ["task_easy", "task_medium", "task_hard"]


def heuristic_agent(obs):

    subject = obs.subject.lower()
    body = obs.body.lower()

    category = "update"
    priority = "medium"
    spam = False

    if "meeting" in subject or "meeting" in body:
        category = "meeting"
        priority = "high"

    elif "security" in subject or "password" in subject:
        category = "work"
        priority = "high"

    elif "free" in subject or "offer" in subject:
        category = "spam"
        priority = "low"
        spam = True

    elif "deadline" in subject or "review" in subject:
        category = "work"
        priority = "high"

    return {
        "category": category,
        "priority": priority,
        "spam": spam
    }


def run_task(task_name):

    env = EmailTriageEnv()
    grader = EmailTriageGrader(task_name)

    obs = env.reset()

    print(f"[START] task={task_name} env=email-triage model={MODEL_NAME}")

    rewards = []
    step = 0

    while True:

        step += 1

        action_dict = heuristic_agent(obs)

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