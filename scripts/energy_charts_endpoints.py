"""List all energy-charts.info endpoints from their OpenAPI spec."""
import requests, json

resp = requests.get("https://api.energy-charts.info/openapi.json", timeout=30)
d = resp.json()
print(f"Title: {d.get('info', {}).get('title')}")
print(f"Version: {d.get('info', {}).get('version')}")
print()
print(f"{'Method':<6} {'Path':<35} Summary")
print("-" * 90)
for path, methods in sorted(d.get("paths", {}).items()):
    for method, meta in methods.items():
        if method in ("get", "post"):
            summary = meta.get("summary", "")
            print(f"{method.upper():<6} {path:<35} {summary}")
