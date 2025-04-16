# app.py
import streamlit as st
from openai_enhanced_chatbot import EnhancedFeedbackChatbot
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

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
st.markdown("- `Provide account issues at Live chat.`")
st.markdown("- `Could you show login issues from Telegram in 2024?`")
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

        # ENHANCED TIME CHARTS 
        if 'charts' in response['data'] and response['data']['charts']:
            st.subheader("Time Analysis")
            for chart_data in response['data']['charts']:
                if chart_data['type'] == 'time_series' and hasattr(chart_data['data'], 'index'):
                    # Get the time series data
                    time_series = chart_data['data']
                    
                    # Ensure we have data spanning more time
                    if len(time_series) > 1:
                        # Make sure the index is datetime
                        try:
                            # Check if index is already datetime
                            if not isinstance(time_series.index, pd.DatetimeIndex):
                                # Try to convert to datetime
                                time_series = pd.Series(
                                    time_series.values,
                                    index=pd.to_datetime(time_series.index)
                                )
                            
                            # Calculate date range
                            date_range = (time_series.index.max() - time_series.index.min()).days
                            
                            # For short date ranges, keep original daily view
                            if date_range < 60:
                                fig, ax = plt.subplots(figsize=(10, 4))
                                ax.plot(time_series.index, time_series.values, marker='o')
                                ax.set_title(f"Messages Over Time{' - ' + chart_data.get('category', '') if 'category' in chart_data else ''}")
                                
                            # For longer ranges, resample to show full year better
                            else:
                                # Create monthly view for ranges over 60 days
                                monthly_data = time_series.resample('M').sum()
                                
                                fig, ax = plt.subplots(figsize=(12, 5))
                                ax.plot(monthly_data.index, monthly_data.values, marker='o', linewidth=2)
                                ax.set_title(f"Monthly Messages Over Time{' - ' + chart_data.get('category', '') if 'category' in chart_data else ''}")
                                
                                # Format x-axis to show month names
                                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                                ax.xaxis.set_major_locator(mdates.MonthLocator())
                                
                                # Add year markers
                                years = sorted(list(set(d.year for d in monthly_data.index)))
                                for year in years:
                                    plt.axvline(pd.Timestamp(f"{year}-01-01"), color='gray', linestyle='--', alpha=0.5)
                                    if monthly_data.max() > 0:  # Check to avoid division by zero
                                        plt.text(pd.Timestamp(f"{year}-01-01"), monthly_data.max() * 1.05, str(year), 
                                                ha='center', va='bottom', fontsize=10)
                                
                                # Calculate and display year-over-year stats if we have multiple years
                                if len(years) > 1 and len(monthly_data) > 12:
                                    yearly_data = time_series.resample('Y').sum()
                                    if len(yearly_data) > 1 and yearly_data.iloc[-2] > 0:  # Prevent division by zero
                                        yoy_change = ((yearly_data.iloc[-1] - yearly_data.iloc[-2]) / yearly_data.iloc[-2]) * 100
                                        change_text = f"Year-over-year change: {yoy_change:.1f}%"
                                        plt.figtext(0.02, 0.02, change_text, fontsize=10, color='blue')
                            
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                        except Exception as e:
                            # Fallback to simple plot if any error occurs with date handling
                            st.write(f"Using simplified time plot due to date formatting: {str(e)}")
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(range(len(time_series)), time_series.values, marker='o')
                            ax.set_title(f"Messages By Time Period{' - ' + chart_data.get('category', '') if 'category' in chart_data else ''}")
                            plt.tight_layout()
                            st.pyplot(fig)

        # ADD MULTI-CATEGORY STATS
        # Only add if there's multi-category data
        if 'multi_category' in response['data'] and response['data']['multi_category']:
            st.subheader("Multi-Category Analysis")
            
            # Get the multi-category results
            multi_cat_results = response['data']['category_results']
            
            # Display simple stats
            total_messages = response['data']['count']
            total_categories = len(multi_cat_results)
            
            # Create metrics
            st.markdown(f"**Messages span across {total_categories} categories**")
            
            # Calculate total categorized messages
            total_categorized = sum(result['count'] for result in multi_cat_results)
            
            # Calculate overlap percentage if more messages are categorized than total
            if total_categorized > total_messages:
                overlap_count = total_categorized - total_messages
                overlap_percent = (overlap_count / total_messages) * 100
                st.info(f"**Category overlap detected:** {overlap_percent:.1f}% of messages could fit in multiple categories ({overlap_count} overlapping categorizations)")
            
            # Show category distribution
            st.markdown("### Category Distribution")
            
            # Create a simple horizontal bar chart with percentages
            fig, ax = plt.subplots(figsize=(10, max(3, len(multi_cat_results) * 0.4)))
            
            # Extract data
            categories = [result['category_name'] for result in multi_cat_results]
            counts = [result['count'] for result in multi_cat_results]
            percentages = [(count/total_messages)*100 for count in counts]
            
            # Sort by count
            sorted_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
            categories = [categories[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]
            percentages = [percentages[i] for i in sorted_indices]
            
            # Plot bars
            y_pos = range(len(categories))
            ax.barh(y_pos, percentages, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('Percentage of Messages')
            ax.set_title('Category Distribution')
            
            # Add percentage labels
            for i, (pct, count) in enumerate(zip(percentages, counts)):
                ax.text(pct + 1, i, f"{pct:.1f}% ({count})", va='center')
            
            plt.tight_layout()
            st.pyplot(fig)

        # Keep the existing category overlap and other sections
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
