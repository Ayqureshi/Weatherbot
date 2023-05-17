import requests
import json
from datetime import datetime, timedelta
import calendar
import time

with open('intents.json', 'r') as file:
    data = json.load(file)

def Average(lst):
    return sum(lst) / len(lst)
    

# Modify the data as needed
now = datetime.now()

def hour_rounder(t):
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
               +timedelta(hours=t.minute//30))

def local_time(tim):
  tim = tim-25200
  ts = int(tim)
  return datetime.utcfromtimestamp(ts).strftime('%H:%M:%S')

now = hour_rounder(now)
time_val = int(time.mktime(now.timetuple()))
time_val = time_val - 25200

response_API = requests.get('https://api.open-meteo.com/v1/forecast?latitude=34.05&longitude=-118.24&hourly=precipitation_probability,precipitation,rain,showers,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high&daily=weathercode,temperature_2m_max,temperature_2m_min,sunrise,sunset,precipitation_sum,precipitation_hours,windspeed_10m_max,windgusts_10m_max&temperature_unit=fahrenheit&timeformat=unixtime&forecast_days=1&timezone=America%2FLos_Angeles')

data_API = response_API.text
parse_json = json.loads(data_API)


active_case = parse_json['hourly']['precipitation_probability']
responses = []

result = parse_json['hourly']['time']
counter = 0
for var in active_case:
  responses.append(f"{var}% at {local_time(result[counter])}")
  counter=counter+1
data["intents"][5]["responses"] = responses

result = parse_json['hourly']['time']
counter = 0
for var in active_case:
  if result[counter] == time_val:
    data["intents"][6]["responses"] = [f"The forecast for Right Now is: {var}%"]
  counter=counter+1
rain_average = Average(active_case)


Temp_max = parse_json['daily']['temperature_2m_max']
data["intents"][7]["responses"] = [f"The forecast for Right Now is: {Temp_max[0]}{chr(176)}"]
Temp_min = parse_json['daily']['temperature_2m_min']
data["intents"][8]["responses"] = [f"The forecast for Right Now is: {Temp_min[0]}{chr(176)}"]

wind_max = parse_json['daily']['windgusts_10m_max']
data["intents"][9]["responses"] = [f"The forecast for today is: {wind_max[0]} km/h"]
wind_min = parse_json['daily']['windspeed_10m_max']
data["intents"][10]["responses"] = [f"The forecast for today is: {wind_min[0]} km/h"]
sunrise = parse_json['daily']['sunrise']
data["intents"][11]["responses"] = [f"Sunrise will be at {local_time(sunrise[0])} AM"]
sunset = parse_json['daily']['sunset']
data["intents"][12]["responses"] = [f"Sunset will be at {local_time(sunset[0])} PM"]
total_precip = parse_json['daily']['precipitation_sum']
data["intents"][13]["responses"] = [f"There will be a total of {total_precip[0]} inches"]

total_precip_hours = parse_json['daily']['precipitation_hours']
data["intents"][14]["responses"] = [f"It will rain for a total of {total_precip_hours[0]} hours"]



acase = parse_json['hourly']['cloudcover']
result = parse_json['hourly']['time']
counter = 0
respons = []
for val in acase:
  respons.append(f"{val}% at {local_time(result[counter])}")
  counter=counter+1 
data["intents"][15]["responses"] = respons   
cloud_average = Average(acase)






rain_status = ""
sunny_status = ""
statis = []


if cloud_average > 40:
  sunny_status = "Cloudy"
if cloud_average > 75:
  sunny_status = "Overcast"
else:
  sunny_status = "Sunny"

statis.append(sunny_status)

if rain_average > 25 :
  rain_status = "Chance of Rain"
if rain_average > 50:
  rain_status = "High Chance of Rain"
if rain_average > 75 and total_precip_hours > 12:
  rain_status = "Rainy"
elif rain_average > 75 and total_precip_hours < 6:
  rain_status = "Showers"

if rain_status != '':
  statis.append(rain_status)
  data["intents"][16]["responses"] = (f"The weather today will be {statis[0]} and {statis[1]}")
else:
  data["intents"][16]["responses"] = (f"The weather today will be {statis[0]}")

with open('intents.json', 'w') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)