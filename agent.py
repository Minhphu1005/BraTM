def get_weather(city):
    import requests
    api_url = f"https://api.weather.com/v1/location/{city}?apikey="
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return data.get("weather", "Khoong cos thong tin thoi tiet")
    else:
        return "Khong the ket noi den API"