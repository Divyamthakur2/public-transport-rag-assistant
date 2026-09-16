#!/usr/bin/env python
# coding: utf-8

# In[11]:


#Import necessary libraries 
import pandas as pd
import os
import re
import openai
import plotly.express as px
import streamlit as st
from datetime import datetime

#LangChain components for chatbot logics
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-proj-example-placeholder-key"

# Load and clean bus route data
df = pd.read_csv("C:\\Divyam\\New data\\Masters Wolverhampton\\Msc Project\\New bus RRoute.csv")
df['Fare'] = df['Fare'].astype(str).str.replace('\u00a3', '', regex=False).astype(float)
df['Departure_Hour'] = pd.to_datetime(df['Departure Time'], errors='coerce').dt.hour
df['Travel_Minutes'] = (
    pd.to_datetime(df['Departure Time'], errors='coerce') -
    pd.to_datetime(df['Arrival Time'], errors='coerce')
).dt.total_seconds() / 60

# Generate semantic text for RAG
df['semantic_text'] = df.apply(lambda row: (
    f"Bus {row['Bus Number']} ({row['Direction']}) stops at {row['Stop Name']} (Order {row['Stop Order']}). "
    f"Arrives: {row['Arrival Time']}, Departs: {row['Departure Time']}, Freq: {row['Frequency']}, "
    f"Day: {row['Day Type']}, Fare: £{row['Fare']:.2f}, "
    f"Connections: {row['Connections'] if pd.notna(row['Connections']) else 'None'}."
), axis=1)

# Vector store setup
loader = DataFrameLoader(df, page_content_column="semantic_text")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
chunks = splitter.split_documents(documents)
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Tool Functions

def get_first_stop(bus_number, direction=None):
    df_filtered = df[df['Bus Number'].astype(str) == str(bus_number)]
    if direction:
        df_filtered = df_filtered[df_filtered['Direction'].str.lower() == direction.lower()]
    if df_filtered.empty:
        return f"No data found for bus {bus_number}."
    first_row = df_filtered.sort_values('Stop Order').iloc[0]
    return f"The first stop of bus {bus_number} ({first_row['Direction']}) is {first_row['Stop Name']}."

def get_last_stop(bus_number, direction=None):
    df_filtered = df[df['Bus Number'].astype(str) == str(bus_number)]
    if direction:
        df_filtered = df_filtered[df_filtered['Direction'].str.lower() == direction.lower()]
    if df_filtered.empty:
        return f"No data found for bus {bus_number}."
    last_row = df_filtered.sort_values('Stop Order', ascending=False).iloc[0]
    return f"The last stop of bus {bus_number} ({last_row['Direction']}) is {last_row['Stop Name']}."

def list_all_stops(bus_number, direction=None):
    df_filtered = df[df['Bus Number'].astype(str) == str(bus_number)]
    if direction:
        df_filtered = df_filtered[df_filtered['Direction'].str.lower() == direction.lower()]
    if df_filtered.empty:
        return f"No route found for bus {bus_number}."
    stops = df_filtered.sort_values('Stop Order')['Stop Name'].unique()
    stops_str = "\n".join([f"- {stop}" for stop in stops])
    return f"Stops for bus {bus_number} ({direction if direction else df_filtered['Direction'].iloc[0]}):\n\n{stops_str}"

def get_buses_through_stop(stop_name):
    df_filtered = df[df['Stop Name'].str.contains(stop_name, case=False, na=False)]
    if df_filtered.empty:
        return f"No buses found passing through {stop_name}."
    buses = df_filtered[['Bus Number', 'Direction']].drop_duplicates()
    bus_list = buses.apply(lambda row: f"- Bus {row['Bus Number']} ({row['Direction']})", axis=1).tolist()
    return f"Buses that pass through {stop_name}:\n" + "\n".join(bus_list)

# Fare Estimator

def estimate_fare_with_routes(start, end):
    try:
        ordered = df.sort_values(['Bus Number', 'Direction', 'Stop Order'])
        routes_start = set(ordered[ordered['Stop Name'].str.contains(start, case=False, na=False)]['Bus Number'])
        routes_end = set(ordered[ordered['Stop Name'].str.contains(end, case=False, na=False)]['Bus Number'])
        common_routes = routes_start & routes_end

        if not common_routes:
            return f"No direct route found between {start} and {end}."

        response_lines = []
        for route in common_routes:
            route_df = ordered[ordered['Bus Number'] == route]
            for direction in route_df['Direction'].unique():
                direction_df = route_df[route_df['Direction'] == direction].sort_values('Stop Order')
                start_rows = direction_df[direction_df['Stop Name'].str.contains(start, case=False)]
                end_rows = direction_df[direction_df['Stop Name'].str.contains(end, case=False)]
                if start_rows.empty or end_rows.empty:
                    continue
                idx_start = start_rows.index[0]
                idx_end = end_rows.index[0]
                segment = direction_df.loc[min(idx_start, idx_end):max(idx_start, idx_end)]
                stop_count = segment['Stop Name'].nunique()
                max_fare = f"£{segment['Fare'].max():.2f}"
                stops_list = segment['Stop Name'].tolist()
                line = f"Bus {route} ({direction}) — From {stops_list[0]} to {stops_list[-1]} — {stop_count} stops — Max Fare: {max_fare}"
                response_lines.append(line)
        return "\n\n".join(response_lines) if response_lines else "No valid route segment found."
    except Exception as e:
        return f"Error: {str(e)}"

def fare_tool_wrapper(text):
    match = re.search(r'(?:fare|go)?\s*from\s+(.*?)\s+to\s+(.*)', text.lower())
    if not match:
        return "Please ask like: 'from [Start] to [End]'"
    start = match.group(1).strip().title()
    end = match.group(2).strip().title()
    return estimate_fare_with_routes(start, end)

# Query Router: Tools + RAG 

def bus_info_tool(user_query: str):
    q = user_query.lower().strip()

    if match := re.search(r'first stop of bus (\d+)', q):
        return get_first_stop(match.group(1))
    elif match := re.search(r'last stop of bus (\d+)', q):
        return get_last_stop(match.group(1))
    elif match := re.search(r'list all stops of bus (\d+)(?: \((inbound|outbound)\))?', q):
        return list_all_stops(match.group(1), match.group(2))
    elif match := re.search(r'(?:buses|which buses) (?:go through|stop at) (.+)', q):
        return get_buses_through_stop(match.group(1).strip())
    elif "from" in q and "to" in q:
        return fare_tool_wrapper(user_query)
    else:
        return qa.run(user_query)

# Streamlit App UI

st.set_page_config(page_title="Smart Bus Chatbot", layout="wide")
st.title("Smart Bus Route Assistant for Stoke-on-Trent")

# Chat + Fare Tabs
chat_tab, fare_tab = st.tabs(["Chat Assistant", "Fare Calculator"])

# Chat Assistant
with chat_tab:
    user_query = st.text_input("Ask your question (e.g., 'First stop of bus 9')")
    if user_query:
        with st.spinner("Searching..."):
            response = bus_info_tool(user_query)
        st.success("Answer:")
        st.write(response)

# Fare Calculator
with fare_tab:
    st.subheader("Estimate fare between two stops")
    col1, col2 = st.columns(2)
    with col1:
        start_stop = st.selectbox("From Stop", sorted(df['Stop Name'].dropna().unique()))
    with col2:
        end_stop = st.selectbox("To Stop", sorted(df['Stop Name'].dropna().unique()))
    if st.button("Estimate Fare"):
        result = estimate_fare_with_routes(start_stop, end_stop)
        st.info(result)

# Sidebar Chart 
if st.sidebar.checkbox(" Show Bus Stop Distribution"):
    stop_counts = df.groupby(['Bus Number', 'Direction'])['Stop Name'].nunique().reset_index(name='Unique Stops')
    fig = px.bar(
        stop_counts,
        x='Bus Number',
        y='Unique Stops',
        color='Direction',
        barmode='group',
        title="Unique Stops per Bus Route"
    )
    st.sidebar.plotly_chart(fig, use_container_width=True)

