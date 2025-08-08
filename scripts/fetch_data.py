import requests
import pandas as pd
import os


# Концепция: Функция для запроса к API. Мы запрашиваем места в Москве.
def fetch_kudago_data(endpoint, params=None):
    base_url = "https://kudago.com/public-api/v1.4/"
    url = base_url + endpoint
    if params is None:
        params = {}
    params['location'] = 'msk'  # Москва
    params['fields'] = (
        'id, title, description, coords, address, images, categories')

    params['lang'] = 'ru'  # Язык
    api_key = os.getenv('KUDAGO_API_KEY')  # Если ключ нужен, иначе удали
    if api_key:
        params['api_key'] = api_key
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Ошибка API: {response.status_code}")
    return response.json()


# Сбор данных о местах и событиях
places_data = fetch_kudago_data(
    "places/", params={'page_size': 100})['results']
events_data = fetch_kudago_data(
    "events/", params={'page_size': 100})['results']

# Обработка: Преобразуем в DataFrame (таблицу)
places_df = pd.DataFrame(places_data)
events_df = pd.DataFrame(events_data)

# Сохранение в JSON для дальнейшего использования
places_df.to_json('data/places.json', orient='records', force_ascii=False)
events_df.to_json('data/events.json', orient='records', force_ascii=False)

print(
    "Данные собраны: ", len(places_df), "мест и ", len(events_df), "событий."
)
