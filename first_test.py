import requests
from sgp4.api import Satrec, jday
from datetime import datetime, timezone

url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
lines = requests.get(url).text.splitlines()

name, line1, line2 = lines[0], lines[1], lines[2]
sat = Satrec.twoline2rv(line1, line2)

now = datetime.now(timezone.utc)
jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)
error, position, velocity = sat.sgp4(jd, fr)

print("Satellite:", name.strip())
print("Position (km):", position)