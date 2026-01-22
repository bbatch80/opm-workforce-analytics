# OPM Workforce Analytics Dashboard

Interactive Streamlit dashboard with embedded AI for analyzing federal workforce data from the Office of Personnel Management (OPM).

## Project Type
Portfolio project demonstrating data visualization, AI integration, and full-stack analytics dashboard development.

## Tech Stack
- **UI Framework:** Streamlit
- **Data Processing:** Pandas, NumPy, PyArrow
- **Visualization:** Plotly
- **AI:** OpenAI GPT-4o-mini (pandas code generation)
- **Data Format:** Parquet (employment), JSONL (accessions/separations)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Streamlit Dashboard (app.py)               │
│  ┌─────────────┐  ┌──────────────────────────────────┐  │
│  │  Sidebar    │  │  Tabs:                           │  │
│  │  - Filters  │  │  [Overview] [Hiring] [Turnover]  │  │
│  │  - Settings │  │  [AI Analyst]                    │  │
│  └─────────────┘  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
    data_loader.py   visualizations.py    query.py
    (caching,        (Plotly charts)      (AI pipeline)
     sampling)                                 │
                                          executor.py
                                          (safe code exec)
```

## Project Structure

```
opm-workforce-analytics/
├── CLAUDE.md              # This file
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── .gitignore
├── app.py                # Main Streamlit dashboard
├── data_loader.py        # Data loading with caching
├── visualizations.py     # Chart generation functions
├── query.py              # AI query pipeline
├── executor.py           # Safe pandas code execution
├── data/                 # Data files (gitignored)
│   ├── employment.parquet
│   ├── accessions.jsonl
│   └── separations.jsonl
└── .streamlit/
    └── secrets.toml.example
```

## Data Sources

| File | Format | Size | Records | Description |
|------|--------|------|---------|-------------|
| `employment.parquet` | Parquet | 32 MB | 2,084,618 | Current workforce snapshot |
| `accessions.jsonl` | JSONL | 16 MB | 13,838 | New hires |
| `separations.jsonl` | JSONL | 24 MB | 20,976 | Departures |

**Key dimensions**: agency, grade, salary, age, education, state, occupation, tenure

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py

# Set API key for AI features
export OPENAI_API_KEY=sk-...
```

## Environment Variables

```
OPENAI_API_KEY=sk-...  # Required for AI Analyst tab
```

## Key Features

### Dashboard Tabs
1. **Overview**: KPIs, workforce by agency, age distribution, salary by grade, geographic distribution
2. **Hiring**: Accession categories, top hiring agencies, education levels, geographic hiring
3. **Turnover**: Separation categories, turnover comparison, tenure analysis, retirement breakdown
4. **AI Analyst**: Natural language questions answered with AI-generated pandas code and visualizations

### AI Code Generation
- User asks question in natural language
- GPT-4o-mini generates pandas code to analyze data
- AST validation ensures code safety (no exec, eval, file access, etc.)
- Code executes in sandbox with 5-second timeout
- Returns visualization + explanation + raw data

### Safety Measures
- AST whitelist validation (blocks dangerous functions)
- Module whitelist (only pandas, numpy, plotly allowed)
- Execution timeout (5 seconds)
- Read-only data operations

## Data Notes

- Large CSV (780MB) converted to Parquet (32MB) for efficient loading
- 10% stratified sample used for interactive queries (memory efficiency)
- Numeric columns pre-converted (salary, tenure)
- REDACTED values converted to NaN

## Deployment

Designed for Streamlit Cloud:
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Add `OPENAI_API_KEY` to Streamlit secrets
4. Data files need to be included or hosted separately
