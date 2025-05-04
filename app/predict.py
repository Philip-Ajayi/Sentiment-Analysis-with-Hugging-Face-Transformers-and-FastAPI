import requests

def get_sentiment(text):
    url = "http://127.0.0.1:8000/predict/"
    response = requests.post(url, json={"text": text})
    result = response.json()
    return result
