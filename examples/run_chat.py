import json
from pathlib import Path

from floranexus_llm.agent import interactive_chat

payload_path = Path(__file__).with_name("sample_prediction.json")
payload = json.loads(payload_path.read_text(encoding="utf-8"))
interactive_chat(payload)
