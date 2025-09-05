import pandas as pd
import requests
from tqdm import tqdm
import os

data = pd.read_csv("../data/main_unions_names_independent_v2.csv") # or afl-cio

list_unions = list(data["main_union"].unique())

subscription_key = "your-subscription-key"
assert subscription_key

search_url = "https://api.bing.microsoft.com/v7.0/search"

unions_to_search = []
for union in list_unions:
    if pd.isnull(union):
        continue
    union = union.replace("/", "_")
    unions_to_search.append(union)

for union in tqdm(unions_to_search):
    try:
        search_term = union + " union facebook"
    except:
        print("Error with search term")
        print(union)
        continue

    union = union.replace("/", "_")

    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": search_term, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()

    # save the search results to a file
    import json
    with open(f"../data/search_results_{union}.json", "w") as f:
        f.write(json.dumps(search_results, indent=4))

