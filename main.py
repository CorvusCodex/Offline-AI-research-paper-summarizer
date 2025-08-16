#!/usr/bin/env python3
"""
Research Paper Summarizer (offline)
Usage:
  python main.py --file paper.txt
"""
import argparse, requests, os, sys, textwrap

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = "llama3.2:4b"
TIMEOUT = 600

def run_llama(prompt):
    r = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("response","").strip()

def build_prompt(paper_text):
    return (
        "You are a research summarizer. Summarize the paper into ~200 words.\n"
        "Include: Problem, Method, Key Results (with metrics if present), Limitations, One takeaway.\n\n"
        f"PAPER:\n{paper_text}\n\nRespond in plain text."
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", "-f")
    args = p.parse_args()
    if not args.file:
        print("Provide --file path to paper text", file=sys.stderr); sys.exit(1)
    try:
        with open(args.file, "r", encoding="utf-8") as fh:
            content = fh.read()
    except Exception as e:
        print("Error reading file:", e, file=sys.stderr); sys.exit(1)
    print(run_llama(build_prompt(content)))

if __name__ == "__main__":
    main()
