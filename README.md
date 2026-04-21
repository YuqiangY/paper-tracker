# Paper Tracker

Daily paper tracking pipeline: arXiv + RSS → keyword filter → LLM scoring → Feishu doc + static HTML.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL
   ```

3. Edit `config.yaml` to customize:
   - Interest areas (keywords, arXiv categories)
   - RSS feed sources
   - Filter thresholds
   - Output settings

## Usage

### Manual run
```bash
python main.py
```

### Crontab (daily at 6:00 AM)
```bash
crontab -e
# Add:
0 6 * * * /path/to/paper-tracker/run.sh >> /path/to/paper-tracker/data/cron.log 2>&1
```

### Feishu output only (retry)
```bash
claude -p "$(sed "s/YYYY-MM-DD/$(date +%F)/g" output/feishu_prompt.md)" --cwd .
```

## Output

- **Daily JSON**: `data/daily/YYYY-MM-DD.json`
- **Static HTML**: `site/index.html`, `site/YYYY-MM-DD.html`
- **Feishu doc**: Created via `claude -p` + mi-feishu MCP
- **SQLite DB**: `data/papers.db` (dedup + history)
