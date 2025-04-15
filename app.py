# app.py

import streamlit as st
from openai_enhanced_chatbot import EnhancedFeedbackChatbot
import os

st.set_page_config(page_title="Feedback Chatbot PETe", layout="wide")

# --- Load the chatbot ---
@st.cache_resource
def load_chatbot():
    # Update the data path to a relative path or a URL
    data_path = "final_dataset_for_chatbot.csv"
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}")
        return None
    return EnhancedFeedbackChatbot(data_path)

chatbot = load_chatbot()

# --- Streamlit UI ---
st.title("PETebot - Feedback Analysis Chatbot")

st.markdown("Ask PETe a question about user feedback. For example:")
st.markdown("- `login issues from Telegram january 2024`")
st.markdown("- `show deposit problems for LiveChat users`")

# User prompt input
query = st.text_input("Ask your question here:")

# Show response
if query and chatbot:
    try:
        response = chatbot.process_query(query)

        # Show main chatbot response
        st.success(response['message'])

        # Show stats
        with st.expander("📊 Stats"):
            st.markdown(f"- **Messages found:** {response['data']['count']}")
            st.markdown(f"- **Unique users:** {response['data']['unique_users']}")

        # Show message samples if available
        if 'messages' in response['data']:
            st.subheader("Sample Messages")
            st.dataframe(response['data']['messages'][['timestamp', 'message', 'predicted_category']])
    except Exception as e:
        st.error(f"Oops! Something went wrong: {e}")
elif query and not chatbot:
    st.error("Chatbot could not be loaded. Please check the dataset path.")
