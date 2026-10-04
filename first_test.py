import requests
from sgp4.api import Satrec, jday
from datetime import datetime, timezone


def fetch_tle_lines(url, timeout=20):
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch TLE data from CelesTrak: {exc}") from exc
    lines = resp.text.splitlines()
    if len(lines) < 3:
        raise RuntimeError("TLE response did not contain enough lines to build a satellite record.")
    return lines


def main():
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    lines = fetch_tle_lines(url)

    name, line1, line2 = lines[0], lines[1], lines[2]
    sat = Satrec.twoline2rv(line1, line2)

    now = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)
    error, position, velocity = sat.sgp4(jd, fr)
    if error != 0:
        raise RuntimeError(f"SGP4 propagation failed with code {error}.")

    print("Satellite:", name.strip())
    print("Position (km):", position)


if __name__ == "__main__":
    main()