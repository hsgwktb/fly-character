import subprocess, time
def sh(c): return subprocess.run(c, shell=True, executable="/bin/bash", capture_output=True, text=True).stdout.strip()
print(sh("tail -3 /content/fly/webui.log"))
print("state1:", sh("curl -s -m 8 http://127.0.0.1:8000/api/state | head -c 320"))
time.sleep(6)
print("state2:", sh("curl -s -m 8 'http://127.0.0.1:8000/api/state?since=0' | python -c \"import sys,json;d=json.load(sys.stdin);print('tick',d['tick'],'t',d['t'],'fps',d['fps'],'acts',d['n_act'],'grounded',d['n_grounded']);print('events',d['events'][-3:])\""))
print("tunnel:", sh("grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /content/tunnel.log | head -1"))
