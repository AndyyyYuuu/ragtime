import os, requests, dotenv

dotenv.load_dotenv()

def retrieve(query: str, api_key: str) -> list[str]:
    response = requests.post("https://apiv2.senso.ai/api/v1/org/search",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        },
        json={
        "query": query,
        "max_results": 10,
        "require_scoped_ids": False
        }
    )
    return response.json()

if __name__ == "__main__":
    query = "Find me the circle-shaped instrument and identify the bar in which it first plays. Avoid saying anything about violins."
    results = retrieve(query, os.environ["SENSO_CSV_KEY"])
    print(results)