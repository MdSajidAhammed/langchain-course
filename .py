import requests
from langsmith import Client

client = Client()


trace_ids = [
    "01a1a4e6-7511-7b49-8de5-371bbad927b7"

]

# LangSmith Project ID
session_id = "9b0558cd-28c2-4cec-998a-556309636838"

url = f"{client.api_url.rstrip('/')}/api/v1/runs/delete"

response = requests.post(
    url,
    headers={
        "x-api-key": client.api_key,
        "Content-Type": "application/json"
    },
    json={
        "trace_ids": trace_ids,
        "session_id": session_id
    },
    timeout=30
)

response.raise_for_status()

print("Trace deletion request submitted!")
print(response.status_code)



# from langsmith import Client

# client = Client()

# projects = list(
#     client.list_projects(
#         name="langchain-course"
#     )
# )

# for project in projects:
#     print("Project Name:", project.name)
#     print("Project ID:", project.id)