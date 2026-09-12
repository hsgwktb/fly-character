import json, os
p = "/content/fly/reflex_result.json"
print(json.dumps(json.load(open(p)), indent=1, ensure_ascii=False) if os.path.exists(p) else "(no result file)")
