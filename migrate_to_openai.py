import os
import glob

# Files to update
agents_dir = "agents"
files = glob.glob(os.path.join(agents_dir, "*.py"))

for file_path in files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replacements
    content = content.replace("from groq import Groq", "from openai import OpenAI")
    content = content.replace('GROQ_API_KEY = os.getenv("GROQ_API_KEY")', 'OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")')
    content = content.replace('GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")', 'OPENAI_MODEL = "gpt-4o-mini"')
    content = content.replace("client = Groq(", "client = OpenAI(")
    content = content.replace("api_key=GROQ_API_KEY", "api_key=OPENAI_API_KEY")
    content = content.replace("model=GROQ_MODEL", "model=OPENAI_MODEL")
    content = content.replace("Groq API error", "OpenAI API error")
    content = content.replace("client = Groq()", "client = OpenAI()")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

print("Replaced Groq with OpenAI in all agents.")
