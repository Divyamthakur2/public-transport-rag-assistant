# Public Transport RAG Assistant

A transport information chatbot developed for my MSc Data Science dissertation at the University of Wolverhampton. The project focuses on bus services in Stoke on Trent and uses Retrieval Augmented Generation with rule based functions to answer questions about routes, stops, timings and fares.

## Project Overview

Public transport information is often spread across timetables, route maps and operator websites. This project explores whether a conversational interface can make that information easier to access.

The application combines a structured bus route dataset with a Retrieval Augmented Generation pipeline. Users can ask questions in natural language, while the system retrieves relevant information from the dataset and generates a response based on that information.

For queries that require exact structured results, such as fares or first and last stops, the application uses rule based functions instead of relying only on the language model.

This project was completed as part of my MSc Data Science dissertation titled *Development of Transport Assistant Chatbot Using RAG for Bus Services in Stoke on Trent*.

## Main Features

| Feature | Description |
|---|---|
| Natural language queries | Users can ask transport related questions in normal English |
| Route and stop information | The chatbot can return route, stop and direction information from the dataset |
| Fare calculator | A separate tool calculates fares between selected stops |
| Semantic retrieval | FAISS and OpenAI embeddings are used to retrieve relevant route information |
| Hybrid query handling | LangChain tools route questions to either structured functions or the RAG pipeline |
| Streamlit interface | The application provides a simple browser based interface for chat and fare queries |
| Data visualisation | Plotly is used to display information about stops and routes |

## Technology Used

| Area | Technology |
|---|---|
| Programming language | Python |
| User interface | Streamlit |
| Data processing | pandas and regular expressions |
| Language model | OpenAI GPT 3.5 Turbo |
| Embeddings | OpenAI text embedding ada 002 |
| Vector search | FAISS |
| Application orchestration | LangChain |
| Visualisation | Plotly Express |

## System Architecture

The application follows a hybrid design.

```text
User Query
    |
    v
Streamlit Interface
    |
    v
LangChain Agent
    |
    +-----------------------------+
    |                             |
    v                             v
Rule Based Functions          RAG Pipeline
    |                             |
    v                             v
Structured Result         Data Preparation
                                  |
                                  v
                           OpenAI Embeddings
                                  |
                                  v
                              FAISS
                                  |
                                  v
                           Relevant Context
                                  |
                                  v
                           GPT 3.5 Turbo
                                  |
                                  v
                              Response
```

The route dataset is converted into natural language text before being indexed. This allows the retrieval system to find relevant records even when the wording of a user question does not exactly match the wording stored in the dataset.

## Dataset

The project uses a manually compiled dataset containing selected bus route information for Stoke on Trent. The dataset was created for academic and demonstration purposes using publicly visible route information as a reference.

The current dataset contains the following fields:

`Bus Number`, `Direction`, `Stop Order`, `Stop Name`, `Arrival Time`, `Departure Time`, `Frequency`, `Day Type`, `Fare`, `Connections`

The dataset is stored in:

```text
data/stoke_bus_routes.csv
```

The data is not intended to be used as an official or live travel information source. Timings and fares may not reflect current operator information.

For current travel information, users should refer to official bus operators or the UK Bus Open Data Service.

## Example Questions

Examples of queries supported by the application include:

```text
What is the last stop of bus 9?

Which buses pass through Shelton?

How can I travel from Hanley to Shelton?

List all stops of bus 21.

What is the first stop of bus 23?
```

The fare calculator also allows users to select an origin and destination from dropdown menus and returns available route information and an estimated fare.

## Evaluation

The system was evaluated using transport related test queries covering route lookup, stop information and fare calculation.

| Metric | Result |
|---|---:|
| Response accuracy | 92.4% |
| Average response time | 0.78 seconds |
| Query classification accuracy | 96% |
| Fare estimation accuracy | 100% |

The main limitations observed during testing were related to vague place descriptions that did not directly match entities in the dataset.

## Repository Structure

```text
public-transport-rag-assistant/
|
|__ app.py
|__ requirements.txt
|__ .env.example
|__ .gitignore
|__ LICENSE
|__ README.md
|
|__ data/
|   |__ stoke_bus_routes.csv
|
|__ docs/
    |__ dissertation.pdf
```

## Installation

Python 3.10 or later is recommended.

Clone the repository:

```bash
git clone https://github.com/Divyamthakur2/public-transport-rag-assistant.git
cd public-transport-rag-assistant
```

Create a virtual environment.

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS or Linux:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## OpenAI API Key

Create a local `.env` file from the provided example file and add your OpenAI API key.

```text
OPENAI_API_KEY=your_api_key_here
```

Do not commit your real API key to GitHub.

## Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

Streamlit will open the application in your browser.

## Dissertation

The full MSc dissertation is available in:

[`docs/dissertation.pdf`](docs/dissertation.pdf)

It contains the literature review, methodology, system design, implementation, evaluation, discussion, limitations and future work for the project.

## Limitations

The current version uses a static dataset and does not connect to live bus locations or delay information.

The chatbot currently supports English text input only.

The dataset covers selected routes and is not a complete representation of the Stoke on Trent bus network.

Vague location descriptions may not always be resolved correctly.

## Future Development

Future improvements could include live transport data, improved location matching, conversational memory, multilingual support, voice input and support for additional transport modes.

## Academic Reference

Thakur, D. (2025). *Development of Transport Assistant Chatbot Using RAG for Bus Services in Stoke on Trent.* MSc Dissertation, University of Wolverhampton.

## Author

Divyam Thakur  
MSc Data Science  
University of Wolverhampton  
2025

Supervisor: Julius Odede

## License

This project is released under the MIT License.
