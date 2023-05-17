import json
import requests
from datetime import datetime, timedelta
import time

# Open the JSON file for reading
with open('intents.json', 'r') as file:
    data = json.load(file)

# Modify the data as needed
now = datetime.now()

def hour_rounder(t):
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
               +timedelta(hours=t.minute//30))

def local_time(tim):
    tim = tim - 25200
    ts = int(tim)
    return datetime.utcfromtimestamp(ts).strftime('%H:%M:%S')

now = hour_rounder(now)
time_val = int(time.mktime(now.timetuple()))
time_val = time_val - 25200

response_API = requests.get('https://api.open-meteo.com/v1/forecast?latitude=34.05&longitude=-118.24&hourly=temperature_2m&temperature_unit=fahrenheit&timeformat=unixtime&forecast_days=1&timezone=America%2FLos_Angeles')

data_API = response_API.text
parse_json = json.loads(data_API)
active_case = parse_json['hourly']['temperature_2m']

result = parse_json['hourly']['time']
counter = 0
responses = []
for var in active_case:
    responses.append(f"{var}{chr(176)} at {local_time(result[counter])}")
    counter += 1
data["intents"][4]["responses"] = responses

result = parse_json['hourly']['time']
counter = 0
for var in active_case:
    if result[counter] == time_val:
        data["intents"][3]["responses"] = [f"The forecast for Right Now is: {var}{chr(176)}"]
        
    counter = counter + 1

# Open the JSON file for writing
with open('intents.json', 'w') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

# # Print the modified response
# print(*data["intents"][3]["responses"], sep="\n")
