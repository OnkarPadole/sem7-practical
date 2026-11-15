#python3 -m pip install requests pandas matplotlib

import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests

API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = "Mumbai"
COUNTRY_CODE = "IN"
CSV_FILE = Path("weather_data.csv")


def get_current_weather(city, country_code, api_key):
    """Get current weather for a city from OpenWeatherMap."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": f"{city},{country_code}",
        "appid": api_key,
        "units": "metric",
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def extract_weather_record(data):
    """Extract the fields used in analysis from the API response."""
    rain = data.get("rain", {})
    snow = data.get("snow", {})

    # OpenWeatherMap reports rain/snow for the preceding 1 or 3 hours,
    # when available. This stores the available amount in millimeters.
    precipitation_mm = rain.get("1h", rain.get("3h", 0))
    precipitation_mm += snow.get("1h", snow.get("3h", 0))

    return {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "city": data["name"],
        "country": data.get("sys", {}).get("country", ""),
        "latitude": data.get("coord", {}).get("lat"),
        "longitude": data.get("coord", {}).get("lon"),
        "temperature_c": data.get("main", {}).get("temp"),
        "humidity_percent": data.get("main", {}).get("humidity"),
        "wind_speed_m_s": data.get("wind", {}).get("speed"),
        "precipitation_mm": precipitation_mm,
        "weather": data.get("weather", [{}])[0].get("description", ""),
    }


def main():
    if not API_KEY:
        raise SystemExit(
            'API key not found. In Terminal run: '
            'export OPENWEATHER_API_KEY="YOUR_API_KEY"'
        )

    try:
        response_data = get_current_weather(CITY, COUNTRY_CODE, API_KEY)
        new_record = extract_weather_record(response_data)
    except requests.exceptions.HTTPError as error:
        raise SystemExit(
            f"OpenWeatherMap returned an error: {error}\n"
            "Check your API key, city name, and account access."
        )
    except requests.exceptions.RequestException as error:
        raise SystemExit(f"Could not connect to OpenWeatherMap: {error}")

    # Append this reading to the dataset, preserving previous readings.
    new_row = pd.DataFrame([new_record])
    if CSV_FILE.exists():
        old_data = pd.read_csv(CSV_FILE)
        weather_data = pd.concat([old_data, new_row], ignore_index=True)
    else:
        weather_data = new_row

    weather_data.to_csv(CSV_FILE, index=False)
    print(f"Saved current reading to {CSV_FILE}")
    print(new_row.to_string(index=False))

    # Convert and clean columns before analysis.
    weather_data["retrieved_at"] = pd.to_datetime(
        weather_data["retrieved_at"], utc=True, errors="coerce"
    )
    numeric_columns = [
        "temperature_c",
        "humidity_percent",
        "wind_speed_m_s",
        "precipitation_mm",
    ]
    for column in numeric_columns:
        weather_data[column] = pd.to_numeric(weather_data[column], errors="coerce")

    weather_data = weather_data.dropna(subset=["retrieved_at", "temperature_c"])
    weather_data = weather_data.drop_duplicates(
        subset=["retrieved_at", "city"]
    ).sort_values("retrieved_at")

    print("\nOverall weather statistics:")
    print(weather_data[numeric_columns].describe())

    # Group readings by day. With frequent readings, this summarizes daily patterns.
    weather_data["day"] = weather_data["retrieved_at"].dt.floor("D")
    daily = weather_data.groupby("day").agg(
        average_temperature_c=("temperature_c", "mean"),
        minimum_temperature_c=("temperature_c", "min"),
        maximum_temperature_c=("temperature_c", "max"),
        average_humidity_percent=("humidity_percent", "mean"),
        average_wind_speed_m_s=("wind_speed_m_s", "mean"),
        total_recorded_precipitation_mm=("precipitation_mm", "sum"),
    )

    print("\nDaily summary (UTC):")
    print(daily)

    daily.to_csv("daily_weather_summary.csv")
    print("\nSaved daily summary to daily_weather_summary.csv")

    # Plot the readings over time.
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(
        weather_data["retrieved_at"],
        weather_data["temperature_c"],
        marker="o",
    )
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title(f"Weather readings for {CITY}")

    axes[1].plot(
        weather_data["retrieved_at"],
        weather_data["humidity_percent"],
        marker="o",
        color="teal",
    )
    axes[1].set_ylabel("Humidity (%)")

    axes[2].bar(
        weather_data["retrieved_at"],
        weather_data["precipitation_mm"],
        width=0.03,
        color="royalblue",
        label="Precipitation",
    )
    axes[2].set_ylabel("Precipitation (mm)")
    axes[2].set_xlabel("Retrieval time (UTC)")

    fig.tight_layout()
    fig.savefig("weather_analysis.png", dpi=150)
    plt.show()

    # The API response includes coordinates; print them for map use.
    print(
        "\nLocation coordinates:",
        f"latitude={new_record['latitude']},",
        f"longitude={new_record['longitude']}",
    )


if __name__ == "__main__":
    main()