# app.py
import streamlit as st
from openai_enhanced_chatbot import EnhancedFeedbackChatbot
import os
import matplotlib.pyplot as plt

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
st.title("PETe - Feedback Analysis Chatbot")

st.markdown("Ask PETe a question about user feedback. For example:")
st.markdown("- `Provide account issues at Live chat in 2024.`")
st.markdown("- `Could you show login issues from Telegram last month?`")
st.markdown("- `show deposit problems for LiveChat users`")
st.markdown("- `what about telegram?`")


# User prompt input
query = st.text_input("Ask your question here:")

# Show response
if query and chatbot:
    try:
        with st.spinner("Analyzing feedback data..."):
            response = chatbot.process_query(query)

        # Show main chatbot response
        st.success(response['message'])

        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Messages Found", response['data']['count'])
        with col2:
            st.metric("Unique Users", response['data']['unique_users'])
        with col3:
            if 'multi_category' in response['data'] and response['data']['multi_category']:
                st.metric("Categories", len(response['data']['category_results']))
            elif 'category_overlap' in response['data'] and response['data']['category_overlap']:
                st.metric("Overlap Rate", f"{response['data']['category_overlap']['overlap_percent']:.1f}%")

        # Display time charts if available
        if 'charts' in response['data'] and response['data']['charts']:
            st.subheader("Time Analysis")
            for chart_data in response['data']['charts']:
                if chart_data['type'] == 'time_series' and hasattr(chart_data['data'], 'index'):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(chart_data['data'].index, chart_data['data'].values, marker='o')
                    ax.set_title(f"Messages Over Time{' - ' + chart_data.get('category', '') if 'category' in chart_data else ''}")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

        # Display category overlap information if available
        if 'category_overlap' in response['data'] and response['data']['category_overlap']:
            overlap = response['data']['category_overlap']
            
            st.subheader("Category Overlap Analysis")
            st.write(f"{overlap['overlap_percent']:.1f}% of messages could fit in multiple categories")
            
            # Create tabs for pairs, examples, and visualization
            overlap_tab1, overlap_tab2 = st.tabs(["Category Pairs", "Example Messages"])
            
            with overlap_tab1:
                # Show top category pairs
                if overlap['top_pairs']:
                    for (cat1, cat2), count in overlap['top_pairs']:
                        cat1_name = chatbot.category_display_names.get(cat1, cat1.replace('_', ' ').title())
                        cat2_name = chatbot.category_display_names.get(cat2, cat2.replace('_', ' ').title())
                        st.write(f"- **{cat1_name}** and **{cat2_name}**: {count} messages")
                
                # Create and display visualization
                if len(overlap['top_pairs']) >= 2:
                    st.subheader("Category Overlap Heatmap")
                    fig = chatbot.create_category_overlap_visualization(
                        chatbot.conversation_memory['last_filtered_df']
                    )
                    if fig:
                        st.pyplot(fig)
            
            with overlap_tab2:
                # Show example messages
                for i, example in enumerate(overlap['sample_overlaps'][:5]):
                    with st.expander(f"Example {i+1}: {example['message'][:50]}..." if len(example['message']) > 50 else example['message']):
                        st.write(f"**Primary category:** {example['primary_category_name']}")
                        
                        # Show potential categories with confidence scores
                        st.write("**Could also fit in:**")
                        for category, score in example['potential_categories']:
                            cat_name = chatbot.category_display_names.get(category, category.replace('_', ' ').title())
                            st.write(f"- {cat_name} ({score:.0f}% confidence)")

        # Show detailed stats
        with st.expander("📊 Detailed Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"- **Total Messages Found:** {response['data']['count']}")
                st.markdown(f"- **Unique Users:** {response['data']['unique_users']}")
                
                if 'multi_category' in response['data'] and response['data']['multi_category']:
                    st.markdown("#### Category Distribution:")
                    for result in response['data']['category_results']:
                        st.markdown(f"- **{result['category_name']}**: {result['count']} messages ({result['count']/response['data']['count']*100:.1f}%)")
            
            with col2:
                # Show temporal insights if available
                if 'temporal_insights' in response['data']:
                    st.markdown("#### Temporal Insights:")
                    for insight in response['data']['temporal_insights']:
                        st.markdown(f"- {insight}")
                        
                # Show user engagement stats if available
                if response['data']['unique_users'] > 0:
                    avg_msgs_per_user = response['data']['count'] / response['data']['unique_users']
                    st.markdown(f"#### User Engagement:")
                    st.markdown(f"- **Avg. Messages per User:** {avg_msgs_per_user:.2f}")

        # Show sample messages if available
        if chatbot.conversation_memory['last_filtered_df'] is not None and len(chatbot.conversation_memory['last_filtered_df']) > 0:
            st.subheader("Sample Messages")
            sample_df = chatbot.conversation_memory['last_filtered_df'].sample(min(5, len(chatbot.conversation_memory['last_filtered_df'])))
            if 'message' in sample_df.columns:
                display_df = sample_df[['timestamp', 'message', 'category']].copy()
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                display_df['category'] = display_df['category'].apply(
                    lambda x: chatbot.category_display_names.get(x, x.replace('_', ' ').title())
                )
                st.dataframe(display_df, use_container_width=True)

    except Exception as e:
        st.error(f"Oops! Something went wrong: {e}")
        st.exception(e)  # This will display the full traceback in development
elif query and not chatbot:
    st.error("Chatbot could not be loaded. Please check the dataset path.")
