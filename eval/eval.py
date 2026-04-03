import os, requests, dotenv

dotenv.load_dotenv()

KEY  = os.environ["SENSO_FULL_KEY"]
BASE = "https://apiv2.senso.ai/api/v1"
HEADERS = {"X-API-Key": KEY, "Content-Type": "application/json"}

resp = requests.post(f"{BASE}/org/kb/folders", headers=HEADERS, json={
    "name": "score_tests",
})
folder = resp.json()
folder_id = folder["kb_node_id"]
print(f"Created folder: {folder['name']} ({folder_id})")