import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()  # Загружаем .env


def fetch_kudago_data(endpoint, params=None):
    base_url = "https://kudago.com/public-api/v1.4/"
    url = base_url + endpoint
    if params is None:
        params = {}
    params["location"] = "msk"
    params["fields"] = "id,title,description,coords,address,images,categories"
    params["lang"] = "ru"
    params["page_size"] = 100
    api_key = os.getenv("KUDAGO_API_KEY")
    if api_key:
        params["api_key"] = api_key
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Ошибка API: {response.status_code}, {response.text}")
    data = response.json()
    print(
        "Получены данные:", [item.keys() for item in data.get("results", [])[:2]]
    )  # Дебаг первых двух записей
    return data.get("results", [])


# Сбор данных
places_data = fetch_kudago_data("places/")
events_data = fetch_kudago_data("events/")

# Проверка данных
if not places_data or not events_data:
    raise ValueError("Нет данных от API")

places_df = pd.DataFrame(places_data)
events_df = pd.DataFrame(events_data)

# Сохранение
os.makedirs("data", exist_ok=True)
places_df.to_json("data/places.json", orient="records", force_ascii=False)
events_df.to_json("data/events.json", orient="records", force_ascii=False)

print(f"Данные собраны: {len(places_df)} мест и {len(events_df)} событий.")
