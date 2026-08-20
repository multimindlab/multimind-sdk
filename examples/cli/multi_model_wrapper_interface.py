import argparse
import os
import subprocess

from dotenv import load_dotenv
from openai import OpenAI


def query_chatgpt(prompt):
    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Please set it in a .env file or as an environment variable.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def query_ollama(prompt):
    result = subprocess.run(
        ["ollama", "run", "mistral", prompt],
        capture_output=True,
        text=True
    )
    return result.stdout

def main():
    parser = argparse.ArgumentParser(description="Query ChatGPT and Mistral using CLI")
    parser.add_argument("--model", choices=["chatgpt", "ollama"], required=True, help="Model to use: 'chatgpt' for OpenAI GPT-4 or 'ollama' for Mistral via Ollama")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt/question to send to the model")
    args = parser.parse_args()

    if args.model == "chatgpt":
        output = query_chatgpt(args.prompt)
    else:
        output = query_ollama(args.prompt)

    print("\n--- Response ---\n")
    print(output)

if __name__ == "__main__":
    main()
