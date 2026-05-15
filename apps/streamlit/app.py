import uuid

import streamlit as st
from dotenv import load_dotenv

from agent.coach import call_coach_with_tools
from persistence.db import ensure_user, load_messages, save_message

load_dotenv()

st.set_page_config(page_title="Running Through Text", page_icon="🏃")
st.title("Running Through Text")
st.caption("Slice 2 — check-in extraction via tool calling")

query_params = st.query_params
if "user_id" in query_params:
    user_id = query_params["user_id"]
elif "user_id" in st.session_state:
    user_id = st.session_state.user_id
else:
    user_id = uuid.uuid4().hex[:8]
    st.session_state.user_id = user_id

ensure_user(user_id)
st.caption(f"Runner: `{user_id}` — bookmark `?user_id={user_id}` to return later")

messages = load_messages(user_id)

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Conta como tá hoje..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    save_message(user_id, "user", prompt)

    history_for_llm = messages + [{"role": "user", "content": prompt}]
    with st.chat_message("assistant"):
        with st.spinner("..."):
            reply = call_coach_with_tools(history_for_llm, user_id=user_id)
        st.markdown(reply)
    save_message(user_id, "assistant", reply)
