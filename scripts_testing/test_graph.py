import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # adds rag-google-io/ to path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env") # load env vars from project root


from agent.graph import run_agent

result = run_agent("What are the trends in agentic AI?")
print(result)