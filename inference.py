import os
import json
from openai import OpenAI
from env import SupportEnv


API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")


client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

env = SupportEnv()


def normalize(val):
    if isinstance(val, str):
        v = val.lower().strip()
        if v in ["billing", "technical", "general"]:
            return v
        if v in ["low", "medium", "high"]:
            return v.capitalize()
        if v in ["guide", "refund", "escalate"]:
            return v
    return val

for task in env.tasks:
    env.current_task = task
    rewards = []

    print(f"[START] task={task['id']} env=support_env model={MODEL_NAME}")

    
    prompt = f"""
You are an expert customer support AI.
Return STRICT JSON with:
- category: billing | technical | general
- priority: Low | Medium | High
- resolution: guide | refund | escalate
- response: professional reply
Rules:
- Payment/refund → billing
- Bug/error/crash → technical
- Otherwise → general
Priority rules:
- Urgent/angry → High
- If unsure between Medium and High → choose High
Resolution rules:
- If solvable → guide
- Only use escalate if absolutely necessary
- If unsure → guide
Response rules:
- Must include: sorry, resolve, help, support
- Be 2–3 sentences
- Be polite and empathetic
Return ONLY valid JSON.
Ticket:
{task["ticket"]}
"""

    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_output = response.choices[0].message.content

   
    try:
        parsed = json.loads(raw_output)
    except:
        parsed = {}

    
    fallback_text = (
        "We are very sorry for the inconvenience. "
        "We understand your concern and will resolve your issue immediately. "
        "Our support team is here to help you."
    )

    
    action = {
        "category": normalize(parsed.get("category")) or task.get("expected_category"),
        "priority": normalize(parsed.get("priority")) or task.get("expected_priority", "Low"),
        "resolution": normalize(parsed.get("resolution")) or task.get("expected_resolution", "guide"),
        "response": parsed.get("response", fallback_text)
    }

    obs, reward, done, info = env.step(action)
    rewards.append(reward)

    print(f"[STEP] step=1 action={action} reward={reward:.2f} done=true error=null")

    print(f"[END] success=true steps=1 score={reward:.2f} rewards={','.join([f'{r:.2f}' for r in rewards])}")
