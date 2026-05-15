import uuid

import streamlit as st
from dotenv import load_dotenv

from agent.coach import (
    call_coach_with_tools,
    maybe_update_coach_note,
    open_wizard,
)
from persistence.db import (
    ensure_user,
    load_messages,
    profile_completeness,
    save_message,
)

load_dotenv()

st.set_page_config(page_title="Running Through Text", page_icon="🏃")
st.title("Running Through Text")
st.caption("Slice 6 — M3 memory + coach_note")

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

# Onboarding progress indicator — visible only while the wizard is active.
# Hides itself once all 5 blocking fields are filled (ADR 0002: same agent,
# this is purely a UI hint, not a separate state machine).
profile_status = profile_completeness(user_id)
if not profile_status["blocking_complete"]:
    st.caption(
        f"Onboarding: {profile_status['filled_count']}/{profile_status['total_count']} fields filled"
    )

messages = load_messages(user_id)

# First-turn wizard opener: if there are no messages and the profile is still
# incomplete, ask the LLM to greet the runner and open the wizard. We pass a
# `system`-role priming message to `open_wizard` rather than a synthetic user
# message — keeps the messages table clean.
if not messages and not profile_status["blocking_complete"]:
    with st.chat_message("assistant"):
        with st.spinner("..."):
            opener = open_wizard(user_id)
        st.markdown(opener)
    save_message(user_id, "assistant", opener)
    messages = [{"role": "assistant", "content": opener}]

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Conta como tá hoje..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    save_message(user_id, "user", prompt)

    history_for_llm = messages + [{"role": "user", "content": prompt}]
    tool_calls_seen: set[str] = set()
    with st.chat_message("assistant"):
        with st.spinner("..."):
            reply = call_coach_with_tools(
                history_for_llm,
                user_id=user_id,
                tool_calls_seen=tool_calls_seen,
            )
        st.markdown(reply)
    save_message(user_id, "assistant", reply)

    # Slice 6: coach-note rewrite happens AFTER the reply renders. We skip
    # it while the wizard is active — onboarding turns have no narrative
    # worth maintaining yet, and the rewrite would just be noise.
    if profile_status["blocking_complete"]:
        maybe_update_coach_note(user_id, tool_calls_seen)
