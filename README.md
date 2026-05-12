# Smart Bus Route Assistant for Stoke-on-Trent

> A Retrieval-Augmented Generation (RAG) chatbot that answers natural-language questions about bus routes, stops, and fares using a structured CSV dataset, no live GPS, no real-time APIs required.

This repository contains the implementation for my **MSc Data Science dissertation** at the University of Wolverhampton (June 2025): *"Development of Transport Assistant Chatbot Using RAG for Bus Services in Stoke-on-Trent."*

---

## Overview

Most public-transport chatbots either rely on expensive real-time GPS feeds or fall back on rigid rule-based menus that break the moment a user phrases a question naturally. This project takes a different approach: it pairs a **static, structured bus-route dataset** with a **hybrid RAG + rule-based pipeline** so users can ask questions like *"How do I get from Hanley to Shelton?"* or *"What's the last stop of bus 9?"* and get accurate, sourced answers in plain English.

The system is designed to be cheap to run, easy to update (swap the CSV, done), and accessible to commuters who struggle with conventional transit apps, particularly older adults and non-native English speakers.

## Key Features

- **Natural-language chat interface** — ask questions in everyday English, no keyword formulas
- **Hybrid query routing** — a LangChain agent decides whether to use the deterministic fare/stop tools or the RAG retrieval pipeline
- **Fare calculator** — dropdown-based fare estimation between any two stops, with stop-count and direction breakdown
- **Semantic search over structured data** — FAISS vector store + OpenAI embeddings let the bot match "Hanley Station" with "Hanley Bus Station" without exact keyword matches
- **Streamlit web UI** — dual-tab layout (Chat Assistant + Fare Calculator) with a sidebar visualisation of stops per route
- **Grounded responses** — answers come from retrieved CSV rows, which keeps hallucinations low

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | OpenAI GPT-3.5-turbo (temperature 0.2) |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Vector store | FAISS (Facebook AI Similarity Search) |
| Orchestration | LangChain (agents, tools, DataFrameLoader) |
| UI | Streamlit |
| Data processing | pandas, regex |
| Charts | Plotly Express |

## Architecture

```
User query (Streamlit UI)
        │
        ▼
LangChain Agent ──► route by intent
        │
        ├──► Rule-based tools (deterministic)
        │       • fare_estimator_tool
        │       • bus_info_tool (first/last stop, all stops, buses-through)
        │
        └──► RAG pipeline (semantic)
                CSV → semantic_text → CharacterTextSplitter
                    → OpenAI embeddings → FAISS index
                    → top-k retrieval → GPT-3.5-turbo → grounded answer
```

The CSV is preprocessed into a `semantic_text` column — one natural-language sentence per row — so each bus stop becomes something like:

> *"Bus 21 (Inbound) stops at Hanley Bus Station (Order 5). Arrives: 06:58, Departs: 07:00, Freq: Every 20m, Day: Weekday, Fare: £2.00, Connections: None."*

This is what gets embedded and retrieved, which is why the bot handles loose phrasing well.

## Dataset

`New_bus_RRoute.csv` is a **manually compiled, illustrative dataset** of bus routes in Stoke-on-Trent, built by the author for academic purposes as part of an MSc dissertation. It was constructed by visually referencing publicly available route information on Google Maps and covers only the major stops along selected routes — it is not a complete or authoritative timetable.

The schema contains the following fields: `Bus Number`, `Direction`, `Stop Order`, `Stop Name`, `Arrival Time`, `Departure Time`, `Frequency`, `Day Type`, `Fare`, `Connections`.

**Source and limitations:**
- Routes, stop names, and approximate timings were recorded by hand from publicly visible map information.
- Fares are illustrative and modelled on a zone-based pricing structure; they do not reflect actual operator prices.
- The dataset is intended solely to demonstrate the RAG chatbot architecture and should **not** be used as a real-world travel reference.
- For accurate, up-to-date Stoke-on-Trent transit data, consult official operator sources (First Potteries, D&G Bus) or the UK Bus Open Data Service ([bus-data.dft.gov.uk](https://www.bus-data.dft.gov.uk/)).

This project is not affiliated with or endorsed by Google, any bus operator, or any transit authority.

## Setup

### Prerequisites
- Python 3.10+
- An OpenAI API key

### Installation

```bash
# clone the repo
git clone https://github.com/<your-username>/stoke-bus-rag-chatbot.git
cd stoke-bus-rag-chatbot

# create a virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate            # Windows

# install dependencies
pip install -r requirements.txt
```

### Configure your API key

Copy the example env file and add your OpenAI key:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your real key:

```
OPENAI_API_KEY=sk-your-key-here
```

`app.py` loads this automatically via `python-dotenv` — never hardcode keys in source files.

### Run it

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Example Queries

**Chat Assistant tab:**
- `What is the last stop of bus 9?`
- `Buses through Shelton`
- `How can I go from Hanley to Shelton?`
- `List all stops of bus 21`
- `First stop of bus 23`

**Fare Calculator tab:** pick a start and end stop from the dropdowns, hit *Estimate Fare*, and the bot returns the bus number, direction, stop count, and max fare for each valid route.

## Results

Evaluated on 50 diverse test queries spanning route lookups, fare calculation, and stop-based questions:

| Metric | Result |
|---|---|
| Response accuracy | 92.4% |
| Average response time | 0.78 s |
| Query classification accuracy | 96% |
| Fare estimation accuracy | 100% (rule-based) |

The 7.6% of queries that missed were mostly ambiguous location references (e.g. *"near the mall"*) where the dataset had no matching entity.

## Repository Structure

```
.
├── app.py                  # main Streamlit app (chatbot + fare calculator)
├── New_bus_RRoute.csv      # Stoke-on-Trent bus route dataset (manually compiled)
├── requirements.txt        # Python dependencies
├── .env.example            # template for your API key (copy to .env)
├── .gitignore              # excludes .env, __pycache__, venv, etc.
├── LICENSE                 # MIT
├── docs/
│   └── dissertation.pdf    # full MSc dissertation
└── README.md
```

## Documentation

The complete MSc dissertation — covering the literature review, methodology, system design, evaluation results, and discussion — is included in [`docs/dissertation.pdf`](docs/dissertation.pdf).

## Limitations

- **No real-time data** — the system uses a static CSV, so it can't account for live delays, diversions, or GPS positions.
- **English-only, text-only** — no voice input or multilingual support in the current build.
- **No conversation memory** — each query is treated independently; follow-up questions need to re-state context.
- **Geographic scope** — currently bounded to the Stoke-on-Trent routes in the CSV; extending to other cities means supplying a new dataset in the same schema.
- **Ambiguity handling** — vague location references ("the mall", "near uni") don't resolve well without fuzzy matching or NER.

## Future Work

- Add fuzzy matching / named entity recognition for messy stop names
- Plug in real-time feeds (GTFS-RT, operator APIs) for live arrivals and delays
- Add conversational memory via LangChain's memory modules for multi-turn dialogue
- Voice input/output and multilingual support to widen accessibility
- Extend the schema to cover rail, tram, and multi-modal journey planning

## Citation

If you reference this work:

> Thakur, D. (2025). *Development of Transport Assistant Chatbot Using RAG for Bus Services in Stoke-on-Trent.* MSc Dissertation, University of Wolverhampton.

## Author

**Divyam Thakur** — MSc Data Science, University of Wolverhampton (2025)
Supervisor: Mr. Julius Odede

## License

Released under the MIT License. See `LICENSE` for details.

## Acknowledgements

Grateful to my supervisor Julius Odede, the University of Wolverhampton, and the open-source teams behind LangChain, FAISS, OpenAI, and Streamlit, whose tools made this project possible.
