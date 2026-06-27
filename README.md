# VA Lottery Pattern Analyzer

Static GitHub Pages dashboard plus a local Python data pipeline for VA Lottery statistical analysis.

## Architecture

- `scripts/scraper.py` gathers lottery result history and writes `data/raw_results.json`.
- `scripts/analyzer.py` reads `data/raw_results.json`, computes statistical analysis, and writes `data/results.json`.
- `index.html` is the GitHub Pages website. It loads `data/results.json` in the browser and displays the dashboard.

## Target games

- Pick 3
- Pick 4
- Cash 5
- Millionaire for Life
- Powerball
- Mega Millions

## Local workflow

Install Python dependencies:

```powershell
pip install requests beautifulsoup4
```

Gather fresh raw results:

```powershell
python scripts\scraper.py
```

Generate website-ready analysis:

```powershell
python scripts\analyzer.py
```

Preview the static site locally:

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8000
```

## GitHub Pages workflow

1. Commit `index.html`, `data/results.json`, `scripts/analyzer.py`, and `scripts/scraper.py`.
2. Push to the GitHub repository for `sheriffllc.github.io`.
3. In GitHub repository settings, enable GitHub Pages from the main branch root.
4. Visit `https://sheriffllc.github.io/`.

## Important disclaimer

This project ranks combinations using historical statistical signals such as frequency, gaps, pair co-occurrence, and position analysis. Lottery drawings are random events. This does not predict future winning numbers.
