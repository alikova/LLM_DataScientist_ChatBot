"""
Enhanced Streamlit interface with better conversation memory display
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from io import BytesIO
import base64

# Try to import the enhanced chatbot
try:
    from openai_enhanced_chatbot import EnhancedFeedbackChatbot
except ImportError:
    from feedback_chatbot import FeedbackAnalysisChatbot as EnhancedFeedbackChatbot

# Page config
st.set_page_config(
    page_title="PETe - Feedback Analysis Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: medium;
        color: #5b5f97;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        gap: 0.75rem;
    }
    .chat-message.user {
        background-color: #e8f4f8;
        border-bottom-right-radius: 0.2rem;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
        border-bottom-left-radius: 0.2rem;
    }
    .chat-message .avatar {
        min-width: 36px;
    }
    .chat-message .avatar img {
        max-width: 36px;
        max-height: 36px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        flex-grow: 1;
        padding-right: 0.5rem;
    }
    .chart-container {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .stats-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-top: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .highlight {
        color: #5b5f97;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state with enhanced memory"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'chatbot' not in st.session_state:
        # Path to your data file - a relative path
        data_path = "final_dataset_for_chatbot.csv"

        # Allow overriding with query parameter
        query_params = st.query_params()
        if "data_path" in query_params:
            data_path = query_params["data_path"][0]

        # Check if the file exists
        if not os.path.exists(data_path):
            st.error(f"Data file not found: {data_path}")
            st.info("Please provide a valid path to your data file.")
            st.stop()

        # Get API key from secrets or environment if available
        openai_api_key = None
        if 'openai_api_key' in st.secrets:
            openai_api_key = st.secrets['openai_api_key']

        # If API key is provided in a text input
        if 'openai_api_key_input' in st.session_state:
            openai_api_key = st.session_state.openai_api_key_input

        try:
            # Initialize the enhanced chatbot
            st.session_state.chatbot = EnhancedFeedbackChatbot(data_path, openai_api_key)
            st.session_state.data_loaded = True

            # Add welcome message
            welcome_message = {
                "role": "assistant",
                "content": "Hello! I'm your User Feedback Analysis assistant PETe Bot with enhanced memory. You can ask me about user feedback and I'll remember our conversation context."
            }
            st.session_state.messages.append(welcome_message)

        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            st.session_state.data_loaded = False


def create_chart(chart_data, chart_type, title=""):
    """Create a chart based on the type and data"""
    fig, ax = plt.subplots(figsize=(10, 4))

    if chart_type == "time_series":
        # Convert index to datetime if it's not already
        if not isinstance(chart_data.index, pd.DatetimeIndex):
            chart_data.index = pd.to_datetime(chart_data.index)

        # Sort by date
        chart_data = chart_data.sort_index()

        # Plot the time series
        ax.plot(chart_data.index, chart_data.values, marker='o', linestyle='-', linewidth=2, markersize=8)
        ax.set_title(title if title else "Message Volume Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Messages")

        # Format x-axis dates
        plt.gcf().autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)

        # Highlight spikes
        mean = chart_data.mean()
        std = chart_data.std()
        threshold = mean + 1.5 * std
        spikes = chart_data[chart_data > threshold]
        if not spikes.empty:
            ax.scatter(spikes.index, spikes.values, color='red', s=100,
                      label=f'Spike(s): {len(spikes)} day(s)', zorder=5)
            ax.legend()

    elif chart_type == "bar":
        # Sort by value
        chart_data = chart_data.sort_values(ascending=False)

        # Plot the bar chart
        ax.bar(chart_data.index, chart_data.values, color='skyblue')
        ax.set_title(title if title else "Distribution")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Add data labels
        for i, v in enumerate(chart_data.values):
            ax.text(i, v + 0.1, str(int(v)), ha='center')

        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add style
    plt.tight_layout()
    return fig

def display_message(message):
    """Display a single message in the chat interface with multi-category support"""
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        st.markdown(content)

        # If there's multi-category data, display it
        if "multi_category_data" in message:
            category_data = message["multi_category_data"]
            
            # Create tabs for each category
            tabs = st.tabs([cat["category_name"] for cat in category_data])
            
            for i, tab in enumerate(tabs):
                with tab:
                    cat_info = category_data[i]
                    
                    # Display category stats
                    st.metric(
                        label="Messages", 
                        value=cat_info["count"],
                        delta=f"{cat_info['unique_users']} users"
                    )
                    
                    # Display confidence score with color
                    confidence = cat_info["confidence"]
                    color = "green" if confidence > 70 else "orange" if confidence > 50 else "red"
                    st.markdown(f"<span style='color:{color}'>Confidence: {confidence:.0f}%</span>", 
                              unsafe_allow_html=True)
                    
                    # If there's sample data for this category
                    if "samples" in cat_info and cat_info["samples"] is not None:
                        with st.expander("View sample messages"):
                            st.dataframe(cat_info["samples"], use_container_width=True)

        # If there's chart data, display it
        if "chart_data" in message:
            for chart in message["chart_data"]:
                st.pyplot(chart)

        # If there's a dataframe, display it
        if "dataframe" in message:
            with st.expander("View sample messages"):
                st.dataframe(message["dataframe"], use_container_width=True)
                
        # If there are category exploration suggestions, show them
        if "category_suggestions" in message:
            with st.expander("Explore related categories"):
                for suggestion in message["category_suggestions"]:
                    if st.button(suggestion, key=f"suggest_{hash(suggestion)}"):
                        # Set this as the next query
                        st.session_state.next_query = suggestion
                        st.rerun()

def display_chat_history():
    """Display the chat history"""
    for message in st.session_state.messages:
        display_message(message)


def process_user_input():
    """Process user input and update chat history"""
    if query := st.chat_input("Ask me about user feedback..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        # Display user message
        with st.chat_message("user"):
            st.markdown(query)

        # Display assistant thinking
        with st.chat_message("assistant"):
            with st.spinner("Analyzing feedback data..."):
                # Process the query
                try:
                    response = st.session_state.chatbot.process_query(query)
                    message = response['message']
                    data = response['data']

                    # Create charts if available
                    charts = []
                    if data['charts']:
                        for chart_data in data['charts']:
                            if chart_data['type'] == 'time_series' and isinstance(chart_data['data'], pd.Series):
                                fig = create_chart(chart_data['data'], 'time_series')
                                charts.append(fig)

                    # Get sample messages
                    if data['count'] > 0:
                        if hasattr(st.session_state.chatbot, 'conversation_memory'):
                            filtered_df = st.session_state.chatbot.conversation_memory['last_filtered_df']
                        else:
                            filtered_df = st.session_state.chatbot.context['last_filtered_df']

                        if filtered_df is not None and len(filtered_df) > 0:
                            sample_df = filtered_df.sample(min(5, len(filtered_df)))
                            sample_df = sample_df[['source', 'timestamp', 'message', 'category']]
                            # Format the timestamp
                            sample_df['timestamp'] = pd.to_datetime(sample_df['timestamp']).dt.strftime('%Y-%m-%d')
                        else:
                            sample_df = None
                    else:
                        sample_df = None

                    # Create response message
                    response_message = {
                        "role": "assistant",
                        "content": message
                    }

                    # Add charts and data if available
                    if charts:
                        response_message["chart_data"] = charts

                    if sample_df is not None:
                        response_message["dataframe"] = sample_df

                    # Display the assistant's response
                    st.markdown(message)

                    # Display charts if available
                    if charts:
                        for chart in charts:
                            st.pyplot(chart)

                    # Display sample data if available
                    if sample_df is not None:
                        with st.expander("View sample messages"):
                            st.dataframe(sample_df, use_container_width=True)

                    # Add response to chat history
                    st.session_state.messages.append(response_message)

                except Exception as e:
                    # In case of an error, display it to the user
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_message)

                    # Add error message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })


def add_sidebar():
    """Add a sidebar with additional information and examples"""
    with st.sidebar:
        st.header("Feedback Analysis Chatbot")
        st.markdown("This chatbot helps you analyze user feedback messages from LiveChat and Telegram.")

        # API Key input
        st.subheader("OpenAI API Key (Optional)")
        if 'openai_api_key_input' not in st.session_state:
            st.session_state.openai_api_key_input = ""

        api_key = st.text_input(
            "Enter your OpenAI API key to enhance query understanding",
            value=st.session_state.openai_api_key_input,
            type="password"
        )

        if api_key != st.session_state.openai_api_key_input:
            st.session_state.openai_api_key_input = api_key
            st.rerun()  # Reload to apply the new API key

        st.subheader("Features")
        st.markdown("""
        -  Filter by category, time range, and source
        -  Get statistical insights
        -  Detect trends and spikes
        -  Conversational memory for follow-up questions
        """)

        st.subheader("Example Queries")
        example_queries = [
            "Show me login issues from Telegram.",
            "How many payment problems were reported in LiveChat?",
            "What about in Telegram?", # Follow-up query
            "Show me bonus activation issues from Live Chat",
            "What are the most common technical errors?",
            "Show me free spins issues"
        ]

        for query in example_queries:
            if st.button(query):
                # Add the example query to the chat input
                # This is a workaround since we can't directly set the chat input value
                st.session_state.messages.append({"role": "user", "content": query})

                # Process the query
                with st.spinner("Processing..."):
                    response = st.session_state.chatbot.process_query(query)
                    message = response['message']
                    data = response['data']

                    # Create response message
                    response_message = {
                        "role": "assistant",
                        "content": message
                    }

                    # Add the response to chat history
                    st.session_state.messages.append(response_message)

                # Force a rerun to update the UI
                st.rerun()

        st.markdown("---")

        # Add a button to reset conversation
        if st.button("Reset Conversation"):
            st.session_state.messages = []

            # Add welcome message back
            welcome_message = {
                "role": "assistant",
                "content": "Hello! I'm your User Feedback Analysis assistant PETe Bot with enhanced memory. You can ask me about user feedback and I'll remember our conversation context."
            }
            st.session_state.messages.append(welcome_message)

            # Reset memory in chatbot
            if hasattr(st.session_state.chatbot, 'conversation_memory'):
                st.session_state.chatbot.conversation_memory = {
                    'messages': [],
                    'last_query': None,
                    'last_filters': {
                        'category': None,
                        'source': None,
                        'time_range': None
                    },
                    'last_filtered_df': None
                }
            else:
                st.session_state.chatbot.context = {
                    'category': None,
                    'source': None,
                    'time_range': None,
                    'last_filtered_df': None
                }

            st.rerun()

        st.caption("Developed for the LLM Data Scientist task")


def main():
    """Main function to run the Streamlit app"""
    # Title
    st.markdown("<div class='main-header'>User Feedback Analysis Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Analyze and gain insights from user feedback messages</div>", unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()

    # Add sidebar
    add_sidebar()

    # Display the chat interface
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        # Display chat history
        display_chat_history()

        # Process user input
        process_user_input()
    else:
        st.error("Failed to load the data. Please check the data path and try again.")


if __name__ == "__main__":
    main()
