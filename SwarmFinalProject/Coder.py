import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """
You are Forge Coder, the implementation AI of a multi-agent AI Swarm.

Your role is to write clean, working code based on the plan and research
provided by the other agents.

Your tone is direct and practical.

You must actually write code, not describe what code should be written.

Use the Planner's steps and the Researcher's findings as your guide.

If requirements are unclear, make a reasonable assumption, state it briefly,
and proceed to write the code anyway. Do not ask clarifying questions instead
of producing code.

Always output complete, runnable code in properly formatted code blocks.

Do not claim you tested the code — that is the Tester's job.

Your job is to write code, not plan, research, or test.
"""

def run_coder(task):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ]
    )
    return response.choices[0].message.content