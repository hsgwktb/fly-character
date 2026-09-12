print(open("/content/llm_status.txt").read())
import os;print("gate:", open("/content/llm_gate.json").read() if os.path.exists("/content/llm_gate.json") else "none")
