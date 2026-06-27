"""
VA Lottery Results Scraper
--------------------------
Fetches historical results for Pick 3, Pick 4, Cash 5, Millionaire for Life,
Powerball, and Mega Millions from the VA Lottery download feed. Each draw is
stored with its main numbers and (when applicable) its bonus ball:
  {"date": "YYYY-MM-DD", "numbers": [...], "bonus": [...], "draw_time": "Day"}

Saves raw data to data/raw_results.json for the analyzer to process.

Usage:
    python scripts/scraper.py
"""

import json
import time
import datetime
import os
import re
import urllib.error
import urllib.request

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw_results.json")

GAME_IDS = {
    "pick3": 1050,
    "pick4": 1040,
    "cash5": 1030,
    "millionaire": 1075,
    "powerball": 20,
    "megamillions": 15,
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; VA-Lottery-Analyzer/1.0)"
}


def normalize_date(value):
    value = value.strip()
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    return value


DATE_RE = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}|[A-Z][a-z]+ \d{1,2}, \d{4}|\d{4}-\d{2}-\d{2}")


def numbers_from_text(value):
    return [int(n) for n in re.findall(r"\d+", value)]


def parse_line(line):
    """
    Parse one result line into one or more draw dicts.

    The VA feed is semicolon-delimited with labeled segments, e.g.:
      Pick 3:  "6/4/2026; Day: 5,9,8; Fireball: 4; Night: 8,6,6; Fireball: 7"
      Powerball: "6/3/2026; 14,16,38,55,64; Powerball: 12"
      Cash 5:  "6/4/2026; 6,12,27,36,37"

    Day/Night become separate draws; any *Ball/Fireball segment is the bonus
    that attaches to the most recent main draw on the line.
    """
    parts = [p.strip() for p in line.split(";") if p.strip()]
    if not parts:
        return []

    date_match = DATE_RE.search(parts[0])
    if not date_match:
        return []
    date = normalize_date(date_match.group(0))

    draws = []
    current = None
    for seg in parts[1:]:
        label, sep, val = seg.partition(":")
        label_l = label.strip().lower()
        body = val if sep else seg

        if "ball" in label_l:                      # Fireball / Powerball / Mega Ball / Millionaire Ball
            bonus = numbers_from_text(body)
            if current is not None:
                current["bonus"] = bonus
        elif label_l in ("day", "night"):          # digit games: two draws per line
            current = {"date": date, "numbers": numbers_from_text(body),
                       "bonus": [], "draw_time": label.strip().title()}
            draws.append(current)
        else:                                       # bare main numbers (e.g. Cash 5)
            nums = numbers_from_text(seg)
            if nums:
                current = {"date": date, "numbers": nums, "bonus": []}
                draws.append(current)

    return [d for d in draws if d["numbers"]]


def parse_download_text(text):
    results = []
    for line in text.splitlines():
        if not line.strip() or not DATE_RE.search(line):
            continue
        results.extend(parse_line(line))
    return results


def fetch_game(game_key, game_id):
    print(f"  Fetching {game_key}...")
    url = f"https://www.valottery.com/api/v1/downloadall?gameId={game_id}"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"  ERROR fetching {game_key}: {e}")
        return []

    results = parse_download_text(text)

    print(f"  Got {len(results)} results for {game_key}")
    return results


def main():
    all_data = {}
    for game_key, game_id in GAME_IDS.items():
        results = fetch_game(game_key, game_id)
        all_data[game_key] = results
        time.sleep(1.5)  # polite delay between requests

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "fetched_at": datetime.datetime.now().isoformat(),
            "games": all_data
        }, f, indent=2)

    print(f"\nSaved raw results -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
