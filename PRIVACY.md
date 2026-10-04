# OrbitSense Privacy Notice

OrbitSense is a local research/educational application and does not require user accounts.

## Data the app retrieves

- Public satellite TLE data from CelesTrak (when network access is available)
- Optional local cache data from `tle_cache.txt` in the project directory
- Optional Earth texture image from the configured public texture URL, with local procedural fallback

## Data the app stores

- Cached TLE text in `tle_cache.txt` for offline fallback and startup resilience

## Analytics and tracking

OrbitSense itself does not intentionally collect user analytics, profiles, or telemetry events.

If you deploy it behind other infrastructure (reverse proxies, hosting platforms, monitoring tools), those systems may collect logs or metadata under their own policies.

## Security note

Do not place secrets in this repository or in local cache files.
