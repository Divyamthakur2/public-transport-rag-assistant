#!/usr/bin/env python
# coding: utf-8

import os
import re

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter

# Keep the original API setup used in the project
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Smart Bus Route Assistant",
    layout="wide"
)


@st.cache_data
def load_bus_data():
    """
    Load and prepare the bus route dataset.
    The original project filename is intentionally retained.
    """
    data = pd.read_csv("New_bus_RRoute.csv")

    # Clean fare values
    data["Fare"] = (
        data["Fare"]
        .astype(str)
        .str.replace("£", "", regex=False)
        .str.strip()
    )
    data["Fare"] = pd.to_numeric(data["Fare"], errors="coerce")

    # Make stop order numeric so route sorting is reliable
    data["Stop Order"] = pd.to_numeric(data["Stop Order"], errors="coerce")

    # Extract hour from departure time
    data["Departure_Hour"] = pd.to_datetime(
        data["Departure Time"],
        format="%H:%M",
        errors="coerce"
    ).dt.hour

    # Calculate the time between arrival and departure
    arrival = pd.to_datetime(
        data["Arrival Time"],
        format="%H:%M",
        errors="coerce"
    )
    departure = pd.to_datetime(
        data["Departure Time"],
        format="%H:%M",
        errors="coerce"
    )

    data["Travel_Minutes"] = (
        departure - arrival
    ).dt.total_seconds() / 60

    # Create natural language text for semantic retrieval
    data["semantic_text"] = data.apply(
        lambda row: (
            f"Bus {row['Bus Number']} ({row['Direction']}) stops at "
            f"{row['Stop Name']} (Order {row['Stop Order']}). "
            f"Arrives: {row['Arrival Time']}, "
            f"Departs: {row['Departure Time']}, "
            f"Frequency: {row['Frequency']}, "
            f"Day: {row['Day Type']}, "
            f"Fare: £{row['Fare']:.2f}, "
            f"Connections: "
            f"{row['Connections'] if pd.notna(row['Connections']) else 'None'}."
        ),
        axis=1
    )

    return data


df = load_bus_data()


@st.cache_resource
def build_rag_system():
    """
    Build the FAISS retrieval system once and reuse it during the session.
    """
    loader = DataFrameLoader(
        df,
        page_content_column="semantic_text"
    )
    documents = loader.load()

    splitter = CharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.2
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False
    )

    return rag_chain


def normalize_bus_number(value):
    """
    Convert values such as 9.0 to 9 so bus numbers compare consistently.
    """
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (ValueError, TypeError):
        pass

    return str(value).strip()


def match_stop_rows(data, stop_name):
    """
    Match a stop using case-insensitive partial text.
    """
    return data[
        data["Stop Name"].astype(str).str.contains(
            re.escape(stop_name),
            case=False,
            na=False
        )
    ]


def estimate_fare_with_routes(start, end):
    """
    Find direct routes serving both stops and calculate the fare for
    the valid travel segment in each direction.
    """
    try:
        ordered = df.sort_values(
            ["Bus Number", "Direction", "Stop Order"]
        ).copy()

        start_matches = match_stop_rows(ordered, start)
        end_matches = match_stop_rows(ordered, end)

        if start_matches.empty:
            return f"No stop matching '{start}' was found."

        if end_matches.empty:
            return f"No stop matching '{end}' was found."

        routes_start = set(
            start_matches["Bus Number"].map(normalize_bus_number)
        )
        routes_end = set(
            end_matches["Bus Number"].map(normalize_bus_number)
        )

        common_routes = routes_start.intersection(routes_end)

        if not common_routes:
            return f"No direct route found between {start} and {end}."

        response_lines = []

        for route in sorted(common_routes):
            route_df = ordered[
                ordered["Bus Number"]
                .map(normalize_bus_number)
                .eq(route)
            ]

            for direction in route_df["Direction"].dropna().unique():
                direction_df = (
                    route_df[
                        route_df["Direction"].eq(direction)
                    ]
                    .sort_values("Stop Order")
                    .reset_index(drop=True)
                )

                start_rows = match_stop_rows(
                    direction_df,
                    start
                )
                end_rows = match_stop_rows(
                    direction_df,
                    end
                )

                if start_rows.empty or end_rows.empty:
                    continue

                start_position = start_rows.index[0]
                end_position = end_rows.index[0]

                # A valid journey must follow the recorded stop order
                if start_position > end_position:
                    continue

                segment = direction_df.iloc[
                    start_position:end_position + 1
                ]

                if segment.empty:
                    continue

                stop_count = segment["Stop Name"].nunique()
                max_fare = segment["Fare"].max()

                if pd.isna(max_fare):
                    fare_text = "Fare unavailable"
                else:
                    fare_text = f"£{max_fare:.2f}"

                first_stop = segment.iloc[0]["Stop Name"]
                last_stop = segment.iloc[-1]["Stop Name"]

                response_lines.append(
                    f"Bus {route} ({direction})\n"
                    f"From: {first_stop}\n"
                    f"To: {last_stop}\n"
                    f"Stops: {stop_count}\n"
                    f"Estimated fare: {fare_text}"
                )

        if not response_lines:
            return (
                f"No valid direct journey was found from "
                f"{start} to {end} in the recorded stop order."
            )

        return "\n\n".join(response_lines)

    except Exception as exc:
        return f"Unable to calculate the fare: {exc}"


def get_route_data(bus_number, direction=None):
    """
    Return route rows for a bus and optional direction.
    """
    bus_number = str(bus_number).strip()

    route_data = df[
        df["Bus Number"]
        .map(normalize_bus_number)
        .eq(bus_number)
    ].copy()

    if direction:
        route_data = route_data[
            route_data["Direction"]
            .astype(str)
            .str.lower()
            .eq(direction.lower())
        ]

    return route_data.sort_values(
        ["Direction", "Stop Order"]
    )


def get_last_stop(bus_number, direction=None):
    route_data = get_route_data(bus_number, direction)

    if route_data.empty:
        return f"No data found for bus {bus_number}."

    if direction:
        last_row = route_data.sort_values(
            "Stop Order",
            ascending=False
        ).iloc[0]

        return (
            f"The last stop of bus {bus_number} "
            f"({last_row['Direction']}) is "
            f"{last_row['Stop Name']}."
        )

    results = []

    for route_direction, group in route_data.groupby("Direction"):
        last_row = group.sort_values(
            "Stop Order",
            ascending=False
        ).iloc[0]

        results.append(
            f"{route_direction}: {last_row['Stop Name']}"
        )

    return (
        f"Last stops for bus {bus_number}:\n\n"
        + "\n".join(results)
    )


def get_first_stop(bus_number, direction=None):
    route_data = get_route_data(bus_number, direction)

    if route_data.empty:
        return f"No data found for bus {bus_number}."

    if direction:
        first_row = route_data.sort_values(
            "Stop Order"
        ).iloc[0]

        return (
            f"The first stop of bus {bus_number} "
            f"({first_row['Direction']}) is "
            f"{first_row['Stop Name']}."
        )

    results = []

    for route_direction, group in route_data.groupby("Direction"):
        first_row = group.sort_values(
            "Stop Order"
        ).iloc[0]

        results.append(
            f"{route_direction}: {first_row['Stop Name']}"
        )

    return (
        f"First stops for bus {bus_number}:\n\n"
        + "\n".join(results)
    )


def format_stop_list(stops):
    return "\n".join(
        f"{index + 1}. {stop}"
        for index, stop in enumerate(stops)
    )


def list_all_stops(bus_number, direction=None):
    route_data = get_route_data(bus_number, direction)

    if route_data.empty:
        return f"No route found for bus {bus_number}."

    if direction:
        stops = (
            route_data
            .sort_values("Stop Order")["Stop Name"]
            .dropna()
            .tolist()
        )

        return (
            f"Stops for bus {bus_number} ({direction.title()}):\n\n"
            f"{format_stop_list(stops)}"
        )

    sections = []

    for route_direction, group in route_data.groupby("Direction"):
        stops = (
            group
            .sort_values("Stop Order")["Stop Name"]
            .dropna()
            .tolist()
        )

        sections.append(
            f"{route_direction}\n"
            f"{format_stop_list(stops)}"
        )

    return (
        f"Stops for bus {bus_number}:\n\n"
        + "\n\n".join(sections)
    )


def get_buses_through_stop(stop_name):
    stop_data = match_stop_rows(df, stop_name)

    if stop_data.empty:
        return (
            f"No buses were found passing through "
            f"{stop_name}."
        )

    buses = (
        stop_data[
            ["Bus Number", "Direction"]
        ]
        .drop_duplicates()
        .copy()
    )

    buses["Bus Number"] = buses[
        "Bus Number"
    ].map(normalize_bus_number)

    buses = buses.sort_values(
        ["Bus Number", "Direction"]
    )

    lines = [
        f"Bus {row['Bus Number']} ({row['Direction']})"
        for _, row in buses.iterrows()
    ]

    return (
        f"Buses that pass through {stop_name}:\n\n"
        + "\n".join(lines)
    )


def bus_info_tool(text):
    """
    Handle structured bus information questions.
    """
    query = text.strip().lower()

    match = re.search(
        r"last stop of bus\s+(\d+)"
        r"(?:\s*\((inbound|outbound)\))?",
        query
    )
    if match:
        return get_last_stop(
            match.group(1),
            match.group(2)
        )

    match = re.search(
        r"first stop of bus\s+(\d+)"
        r"(?:\s*\((inbound|outbound)\))?",
        query
    )
    if match:
        return get_first_stop(
            match.group(1),
            match.group(2)
        )

    match = re.search(
        r"(?:list\s+)?all stops of bus\s+(\d+)"
        r"(?:\s*\((inbound|outbound)\))?",
        query
    )
    if match:
        return list_all_stops(
            match.group(1),
            match.group(2)
        )

    match = re.search(
        r"(?:which\s+)?buses?\s+"
        r"(?:go\s+through|pass\s+through|through|stop\s+at)\s+(.+)",
        query
    )
    if match:
        return get_buses_through_stop(
            match.group(1).strip()
        )

    return None


def fare_tool_wrapper(text):
    """
    Extract origin and destination from natural language.
    """
    match = re.search(
        r"(?:fare\s+)?(?:from\s+)?"
        r"(.+?)\s+to\s+(.+)",
        text.strip(),
        flags=re.IGNORECASE
    )

    if not match:
        return None

    start = match.group(1).strip()
    end = match.group(2).strip()

    # Remove common introductory phrases from the origin
    start = re.sub(
        r"^(?:how can i go|how do i get|how can i travel|travel|go)\s+",
        "",
        start,
        flags=re.IGNORECASE
    ).strip()

    return estimate_fare_with_routes(start, end)


def looks_like_fare_query(text):
    query = text.lower()

    return (
        " to " in query
        and (
            "from " in query
            or "fare" in query
            or "how can i go" in query
            or "how do i get" in query
            or "how can i travel" in query
        )
    )


def answer_query(user_query):
    """
    Use deterministic functions where possible, then use RAG
    for general transport questions.
    """
    structured_answer = bus_info_tool(user_query)

    if structured_answer is not None:
        return structured_answer

    if looks_like_fare_query(user_query):
        fare_answer = fare_tool_wrapper(user_query)
        if fare_answer is not None:
            return fare_answer

    try:
        rag_chain = build_rag_system()
        response = rag_chain.run(user_query)

        if response:
            return response

        return "I could not find enough information in the bus route data."

    except Exception as exc:
        return (
            "The transport information could not be retrieved. "
            f"Details: {exc}"
        )


# Application interface
st.title("Smart Bus Route Assistant for Stoke-on-Trent")

st.write(
    "Ask questions about bus routes, stops and fares using "
    "the transport dataset used in this project."
)

chat_tab, fare_tab = st.tabs(
    ["Chat Assistant", "Fare Calculator"]
)


with chat_tab:
    st.subheader("Chat Assistant")

    user_query = st.text_input(
        "Enter your question",
        placeholder=(
            "For example: What is the last stop of bus 9?"
        )
    )

    if user_query:
        with st.spinner("Searching the bus information..."):
            response = answer_query(user_query)

        st.markdown("### Answer")
        st.write(response)


with fare_tab:
    st.subheader("Fare Calculator")

    available_stops = sorted(
        df["Stop Name"]
        .dropna()
        .astype(str)
        .unique()
    )

    col1, col2 = st.columns(2)

    with col1:
        start_stop = st.selectbox(
            "From Stop",
            available_stops
        )

    with col2:
        end_stop = st.selectbox(
            "To Stop",
            available_stops
        )

    if st.button(
        "Estimate Fare",
        type="primary"
    ):
        if start_stop == end_stop:
            st.warning(
                "Please select two different stops."
            )
        else:
            result = estimate_fare_with_routes(
                start_stop,
                end_stop
            )
            st.info(result)


# Sidebar chart
st.sidebar.header("Route Overview")

if st.sidebar.checkbox(
    "Show Bus Stop Distribution"
):
    stop_counts = (
        df.groupby(
            ["Bus Number", "Direction"]
        )["Stop Name"]
        .nunique()
        .reset_index(name="Unique Stops")
    )

    fig = px.bar(
        stop_counts,
        x="Bus Number",
        y="Unique Stops",
        color="Direction",
        barmode="group",
        title="Unique Stops per Bus Route"
    )

    st.sidebar.plotly_chart(
        fig,
        use_container_width=True
    )
