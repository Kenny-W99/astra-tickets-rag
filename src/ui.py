from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="AstraTickets RAG", layout="wide")

st.title("AstraTickets — Support RAG Demo")

api_url = st.sidebar.text_input("API URL", "http://localhost:8000")

query = st.text_area("Ask a question", "What is the refund policy for cancelled bookings?")

if st.button("Ask"):
    with st.spinner("Thinking..."):
        r = requests.post(f"{api_url}/chat", json={"query": query}, timeout=60)
        r.raise_for_status()
        data = r.json()

    st.subheader("Answer")
    st.write(data["answer"])

    st.subheader("Retrieved context")
    for i, c in enumerate(data.get("contexts", []), start=1):
        with st.expander(f"{i}. Ticket {c.get('doc_id')} (score {c.get('score'):.3f})"):
            st.write(c.get("text", ""))
