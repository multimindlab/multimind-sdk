import openai
import subprocess
import argparse

openai.api_key = "your_openai_api_key_here"

def query_chatgpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content']

def query_ollama(prompt):
    result = subprocess.run(
        ["ollama", "run", "mistral", prompt],
        capture_output=True,
        text=True
    )
    return result.stdout

def main():
    parser = argparse.ArgumentParser(description="Query ChatGPT and Mistral using CLI")
    parser.add_argument("--model", choices=["chatgpt", "ollama"], required=True)
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()

    if args.model == "chatgpt":
        output = query_chatgpt(args.prompt)
    else:
        output = query_ollama(args.prompt)

    print("\n--- Response ---\n")
    print(output)

if __name__ == "__main__":
    main() 