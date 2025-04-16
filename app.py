# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from openai_enhanced_chatbot import EnhancedFeedbackChatbot
import os
import calendar
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Feedback Chatbot PETe", layout="wide", initial_sidebar_state="expanded")

# Apply custom styling
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; margin-bottom: 1rem;}
    .sub-header {font-size: 1.5rem; font-weight: medium; color: #5b5f97; margin-bottom: 1rem;}
    .stats-box {padding: 1rem; border-radius: 0.5rem; background-color: #f8f9fa; 
               margin-top: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);}
    .chart-container {background-color: white; border-radius: 0.5rem; padding: 1rem; 
                     margin-top: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);}
    .highlight {color: #5b5f97; font-weight: bold;}
    .stButton>button {width: 100%;}
</style>
""", unsafe_allow_html=True)

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

# Function to create time series chart with enhanced features
def create_time_series_chart(data_series, title="Messages Over Time", show_trend=True, 
                             show_spikes=True, resample_freq=None):
    """
    Create an enhanced time series visualization with trend lines and spike detection
    
    Args:
        data_series: Pandas series with datetime index and count values
        title: Chart title
        show_trend: Whether to show trend line
        show_spikes: Whether to highlight spikes
        resample_freq: Frequency to resample data (None, 'W', 'M', 'Q', 'Y')
    
    Returns:
        Plotly figure object
    """
    # Make sure we have a sorted time series
    data_series = data_series.sort_index()
    
    # Resample data if frequency is specified
    if resample_freq:
        # Map string to actual frequency
        freq_map = {
            'W': 'Weekly',
            'M': 'Monthly', 
            'Q': 'Quarterly',
            'Y': 'Yearly'
        }
        
        # Resample the data
        data_series = data_series.resample(resample_freq).sum()
        freq_name = freq_map.get(resample_freq, 'Resampled')
        title = f"{title} ({freq_name})"
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add main time series line
    fig.add_trace(go.Scatter(
        x=data_series.index, 
        y=data_series.values,
        mode='lines+markers',
        name='Message Count',
        line=dict(color='royalblue', width=2),
        marker=dict(size=8),
        hovertemplate='%{x}<br>Messages: %{y}<extra></extra>'
    ))
    
    # Add trend line if requested
    if show_trend and len(data_series) > 3:
        # Create numeric x values for trend calculation
        x_numeric = np.arange(len(data_series))
        y_values = data_series.values
        
        # Calculate trend line using numpy's polyfit
        trend_coeffs = np.polyfit(x_numeric, y_values, 1)
        trend_line = np.polyval(trend_coeffs, x_numeric)
        
        # Calculate trend percentage change
        if trend_line[0] != 0:
            trend_change = ((trend_line[-1] - trend_line[0]) / trend_line[0]) * 100
            trend_direction = "increasing" if trend_change > 0 else "decreasing"
            trend_label = f"{abs(trend_change):.1f}% {trend_direction} trend"
        else:
            trend_label = "Trend line"
        
        # Add trend line to plot
        fig.add_trace(go.Scatter(
            x=data_series.index,
            y=trend_line,
            mode='lines',
            name=trend_label,
            line=dict(color='rgba(255, 127, 14, 0.7)', width=2, dash='dash'),
            hovertemplate='%{x}<br>Trend: %{y:.1f}<extra></extra>'
        ))
    
    # Detect and highlight spikes
    if show_spikes and len(data_series) > 3:
        mean = data_series.mean()
        std = data_series.std()
        threshold = mean + 1.5 * std
        
        spikes = data_series[data_series > threshold]
        if not spikes.empty:
            fig.add_trace(go.Scatter(
                x=spikes.index,
                y=spikes.values,
                mode='markers',
                marker=dict(size=12, color='red', symbol='star'),
                name=f'Spikes ({len(spikes)} detected)',
                hovertemplate='%{x}<br>Spike: %{y} messages<extra></extra>'
            ))
    
    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Number of Messages',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Improve date formatting on x-axis
    date_range = (data_series.index.max() - data_series.index.min()).days
    
    if date_range > 365:
        # For year-long data, show quarters
        fig.update_xaxes(
            tickformat="%b %Y",
            tickmode='array',
            tickvals=pd.date_range(start=data_series.index.min(), 
                                   end=data_series.index.max(), 
                                   freq='Q')
        )
    elif date_range > 60:
        # For month-long data, show months
        fig.update_xaxes(
            tickformat="%b %d",
            tickmode='array',
            tickvals=pd.date_range(start=data_series.index.min(), 
                                   end=data_series.index.max(), 
                                   freq='M')
        )
    
    return fig

# Function to create calendar heatmap
def create_calendar_heatmap(data_series, year=None):
    """
    Create a calendar heatmap visualization showing message volume by day
    
    Args:
        data_series: Pandas series with datetime index and count values
        year: Specific year to display (defaults to most recent)
        
    Returns:
        Plotly figure object
    """
    # Make sure we have date index
    data_series = data_series.sort_index()
    
    # Get year to display
    if year is None:
        year = data_series.index.max().year
    
    # Filter data for the specified year
    year_data = data_series[data_series.index.year == year]
    
    if year_data.empty:
        return None
    
    # Create a DataFrame with all days in the year
    all_days = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
    full_year_df = pd.DataFrame(index=all_days)
    
    # Merge with actual data
    merged_data = full_year_df.join(pd.DataFrame({'count': year_data})).fillna(0)
    merged_data['count'] = merged_data['count'].astype(int)
    
    # Extract day, month, weekday
    merged_data['day'] = merged_data.index.day
    merged_data['month'] = merged_data.index.month
    merged_data['weekday'] = merged_data.index.weekday
    
    # Create month labels
    month_labels = [calendar.month_abbr[i] for i in range(1, 13)]
    weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # Create the heatmap using plotly
    fig = px.imshow(
        merged_data.pivot(index='weekday', columns='month', values='count').fillna(0),
        labels=dict(x="Month", y="Day of Week", color="Messages"),
        x=month_labels,
        y=weekday_labels,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        title=f"Message Volume Heatmap for {year}",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# Function to create multi-source comparison chart
def create_source_comparison(data, sources, time_range=None):
    """
    Create a comparison chart between different sources over time
    
    Args:
        data: Filtered dataframe with timestamp and source columns
        sources: List of source names to compare
        time_range: Optional tuple of (start_date, end_date)
        
    Returns:
        Plotly figure object
    """
    # Filter by time range if provided
    if time_range:
        start_time, end_time = time_range
        data = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]
    
    # Create pivot table by date and source
    data['date'] = data['timestamp'].dt.date
    pivot_data = data.pivot_table(
        index='date', 
        columns='source', 
        values='message',
        aggfunc='count',
        fill_value=0
    )
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add a line for each source
    for source in sources:
        if source in pivot_data.columns:
            # Resample to smooth very sparse data
            date_range = (pivot_data.index.max() - pivot_data.index.min()).days
            
            # Determine appropriate resampling frequency based on date range
            if date_range > 180:
                # For longer periods, use weekly resampling
                resample_freq = '7D'
            else:
                # For shorter periods, use 3-day resampling
                resample_freq = '3D'
                
            # Convert index to datetime and resample
            source_data = pd.Series(
                pivot_data[source].values, 
                index=pd.to_datetime(pivot_data.index)
            ).resample(resample_freq).mean().fillna(0)
            
            # Add to plot
            fig.add_trace(go.Scatter(
                x=source_data.index,
                y=source_data.values,
                mode='lines+markers',
                name=source.title(),
                hovertemplate='%{x}<br>%{y:.1f} messages<extra></extra>'
            ))
    
    # Customize layout
    fig.update_layout(
        title="Message Volume by Source",
        xaxis_title="Date",
        yaxis_title="Number of Messages",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# Function to create a category distribution visualization
def create_category_distribution(data, top_n=8):
    """
    Create a visualization of the category distribution
    
    Args:
        data: Filtered dataframe with category column
        top_n: Number of top categories to display
        
    Returns:
        Plotly figure object
    """
    # Count messages by category
    category_counts = data['category'].value_counts()
    
    # Get display names
    display_names = {}
    for cat in category_counts.index:
        display_names[cat] = chatbot.category_display_names.get(
            cat, cat.replace('_', ' ').title()
        )
    
    # Limit to top N categories
    if len(category_counts) > top_n:
        top_categories = category_counts.head(top_n-1)
        other_count = category_counts[top_n-1:].sum()
        
        # Create a new series with top categories and "Other"
        updated_counts = pd.Series({
            **{display_names[cat]: count for cat, count in top_categories.items()},
            "Other": other_count
        })
    else:
        # Use all categories with display names
        updated_counts = pd.Series({
            display_names[cat]: count for cat, count in category_counts.items()
        })
    
    # Create plotly figure with two visualizations
    fig = make_subplots(
        rows=1, 
        cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=("Percentage Distribution", "Message Counts")
    )
    
    # Add pie chart
    fig.add_trace(
        go.Pie(
            labels=updated_counts.index,
            values=updated_counts.values,
            hole=0.4,
            textinfo="percent+label",
            insidetextorientation="radial",
            hovertemplate="%{label}<br>%{value} messages (%{percent})<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=updated_counts.index,
            y=updated_counts.values,
            text=updated_counts.values,
            textposition="auto",
            hovertemplate="%{x}<br>%{y} messages<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Category Distribution",
        showlegend=False,
        height=450,
        margin=dict(l=20, r=20, t=60, b=60)
    )
    
    # Update y-axis title for bar chart
    fig.update_yaxes(title_text="Number of Messages", row=1, col=2)
    
    # Update x-axis in bar chart to show angled labels
    fig.update_xaxes(tickangle=45, row=1, col=2)
    
    return fig

# Function to visualize seasonal patterns
def create_seasonal_analysis(data):
    """
    Create visualizations showing seasonal patterns in the data
    
    Args:
        data: Filtered dataframe with timestamp column
        
    Returns:
        Tuple of (weekday_fig, hourly_fig, monthly_fig)
    """
    # Extract temporal components
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['month'] = data['timestamp'].dt.month
    
    # Create figures for each temporal aspect
    
    # 1. Messages by day of week
    weekday_counts = data.groupby('day_of_week').size()
    
    # Reindex to ensure all weekdays are included
    days = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
            3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    weekday_counts = weekday_counts.reindex(range(7), fill_value=0)
    
    weekday_fig = px.bar(
        x=[days[i] for i in range(7)],
        y=weekday_counts.values,
        labels={'x': 'Day of Week', 'y': 'Number of Messages'},
        title="Messages by Day of Week",
        color=weekday_counts.values,
        color_continuous_scale='Viridis'
    )
    
    weekday_fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False
    )
    
    # 2. Messages by hour of day
    hourly_counts = data.groupby('hour').size()
    
    # Reindex to ensure all hours are included
    hourly_counts = hourly_counts.reindex(range(24), fill_value=0)
    
    hourly_fig = px.bar(
        x=[f"{h:02d}:00" for h in range(24)],
        y=hourly_counts.values,
        labels={'x': 'Hour of Day', 'y': 'Number of Messages'},
        title="Messages by Hour of Day",
        color=hourly_counts.values,
        color_continuous_scale='Viridis'
    )
    
    hourly_fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False,
        xaxis_tickmode='array',
        xaxis_tickvals=[f"{h:02d}:00" for h in range(0, 24, 3)]
    )
    
    # 3. Messages by month
    monthly_counts = data.groupby('month').size()
    
    # Reindex to ensure all months are included
    months = {i: calendar.month_abbr[i] for i in range(1, 13)}
    monthly_counts = monthly_counts.reindex(range(1, 13), fill_value=0)
    
    monthly_fig = px.bar(
        x=[months[i] for i in range(1, 13)],
        y=monthly_counts.values,
        labels={'x': 'Month', 'y': 'Number of Messages'},
        title="Messages by Month",
        color=monthly_counts.values,
        color_continuous_scale='Viridis'
    )
    
    monthly_fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False
    )
    
    return weekday_fig, hourly_fig, monthly_fig

# --- Sidebar with filters ---
def add_sidebar():
    with st.sidebar:
        st.markdown("<div class='sub-header'>PETe Bot Controls</div>", unsafe_allow_html=True)
        
        # Optional: Add date range filter
        if chatbot and hasattr(chatbot, 'df'):
            min_date = chatbot.df['timestamp'].min().date()
            max_date = chatbot.df['timestamp'].max().date()
            
            st.subheader("Date Range Filter")
            date_range = st.date_input(
                "Select date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            # Example queries based on selected date range
            if len(date_range) == 2:
                start_date, end_date = date_range
                st.subheader("Example Queries")
                
                example_queries = [
                    f"login issues between {start_date} and {end_date}",
                    "payment problems in LiveChat",
                    "compare bonus issues between Telegram and LiveChat",
                    "show me trends for technical errors",
                    "free spins issues by month"
                ]
                
                for query in example_queries:
                    if st.button(query):
                        st.session_state.query = query
                        st.rerun()
        
        # Advanced visualization options
        st.subheader("Visualization Options")
        
        st.session_state.show_calendar = st.checkbox("Show Calendar Heatmap", value=True)
        st.session_state.show_trends = st.checkbox("Show Trend Analysis", value=True)
        st.session_state.show_seasonal = st.checkbox("Show Seasonal Patterns", value=True)
        
        time_options = {
            'raw': 'Original',
            'W': 'Weekly', 
            'M': 'Monthly',
            'Q': 'Quarterly'
        }
        
        st.session_state.time_scale = st.selectbox(
            "Time Scale",
            options=list(time_options.keys()),
            format_func=lambda x: time_options[x],
            index=1  # Default to weekly
        )
        
        # Reset button
        if st.button("Reset Analysis", use_container_width=True):
            st.session_state.query = ""
            st.rerun()

# --- Main Application UI ---
def main():
    # Add sidebar
    add_sidebar()
    
    # Header
    st.markdown("<div class='main-header'>PETe - Feedback Analysis Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Analyze and gain insights from user feedback messages</div>", unsafe_allow_html=True)
    
    # Instruction area
    with st.expander("📋 How to use PETe"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Example Queries")
            st.markdown("- `Show login issues from LiveChat in January 2024`")
            st.markdown("- `Compare payment problems between Telegram and LiveChat`")
            st.markdown("- `What bonus issues were reported in the last 3 months?`")
            st.markdown("- `Show me free spins issues by month`")
        
        with col2:
            st.markdown("### Tips")
            st.markdown("- You can ask follow-up questions like `what about Telegram?`")
            st.markdown("- Use the sidebar to filter by date range")
            st.markdown("- Try comparing different sources or categories")
            st.markdown("- Ask for specific time periods or trends")
    
    # User prompt input - use session state to maintain value across reruns
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    query = st.text_input("Ask your question here:", value=st.session_state.query)
    
    # Store query in session state
    if query != st.session_state.query:
        st.session_state.query = query
    
    # Show response
    if query and chatbot:
        try:
            with st.spinner("Analyzing feedback data..."):
                response = chatbot.process_query(query)
            
            # --- Main response area ---
            st.markdown(f"<div class='stats-box'>{response['message']}</div>", unsafe_allow_html=True)
            
            # --- Key metrics section ---
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Messages Found", response['data']['count'])
            with col2:
                st.metric("Unique Users", response['data']['unique_users'])
            with col3:
                # Calculate messages per day
                if chatbot.conversation_memory['last_filtered_df'] is not None:
                    filtered_df = chatbot.conversation_memory['last_filtered_df']
                    if len(filtered_df) > 0:
                        date_range = (filtered_df['timestamp'].max() - filtered_df['timestamp'].min()).days
                        msgs_per_day = len(filtered_df) / max(1, date_range)
                        st.metric("Avg. Messages/Day", f"{msgs_per_day:.1f}")
            with col4:
                # Show multi-category or category overlap info
                if 'multi_category' in response['data'] and response['data']['multi_category']:
                    st.metric("Categories", len(response['data']['category_results']))
                elif 'category_overlap' in response['data'] and response['data']['category_overlap']:
                    st.metric("Overlap Rate", f"{response['data']['category_overlap']['overlap_percent']:.1f}%")
                elif chatbot.conversation_memory['last_filtered_df'] is not None:
                    # Calculate user engagement
                    filtered_df = chatbot.conversation_memory['last_filtered_df']
                    if len(filtered_df) > 0 and response['data']['unique_users'] > 0:
                        msgs_per_user = len(filtered_df) / response['data']['unique_users']
                        st.metric("Msgs/User", f"{msgs_per_user:.1f}")
            
            # --- Enhanced Time Series Visualization ---
            if chatbot.conversation_memory['last_filtered_df'] is not None:
                filtered_df = chatbot.conversation_memory['last_filtered_df']
                
                if len(filtered_df) >= 5:
                    # Create time series data - group by day
                    filtered_df['date'] = filtered_df['timestamp'].dt.date
                    daily_counts = filtered_df.groupby('date').size()
                    
                    # Time series title based on filters
                    series_title = "Messages Over Time"
                    if 'category' in chatbot.conversation_memory['current_context'] and chatbot.conversation_memory['current_context']['category']:
                        category = chatbot.conversation_memory['current_context']['category']
                        category_name = chatbot.category_display_names.get(
                            category, category.replace('_', ' ').title()
                        )
                        series_title = f"{category_name} Messages Over Time"
                    
                    # Create time series chart with proper scaling
                    time_series_fig = create_time_series_chart(
                        daily_counts, 
                        title=series_title,
                        show_trend=st.session_state.show_trends,
                        show_spikes=True,
                        resample_freq=st.session_state.time_scale if st.session_state.time_scale != 'raw' else None
                    )
                    
                    # Display the time series
                    st.plotly_chart(time_series_fig, use_container_width=True)
                    
                    # --- Calendar Heatmap ---
                    if st.session_state.show_calendar and len(filtered_df) >= 15:
                        # Get the years in the data
                        years = sorted(filtered_df['timestamp'].dt.year.unique())
                        
                        if years:
                            # Default to the latest year
                            latest_year = years[-1]
                            
                            # Allow selecting a year if there are multiple
                            if len(years) > 1:
                                selected_year = st.selectbox(
                                    "Select year for calendar view:",
                                    options=years,
                                    index=len(years)-1
                                )
                            else:
                                selected_year = latest_year
                            
                            # Create the calendar heatmap
                            calendar_fig = create_calendar_heatmap(daily_counts, year=selected_year)
                            
                            if calendar_fig:
                                st.plotly_chart(calendar_fig, use_container_width=True)
                    
                    # --- Create tabs for detailed analysis ---
                    tabs = st.tabs([
                        "📊 Category Breakdown", 
                        "🔄 Source Comparison", 
                        "📅 Seasonal Patterns"
                    ])
                    
                    # Tab 1: Category Breakdown
                    with tabs[0]:
                        if 'category' in filtered_df.columns:
                            category_fig = create_category_distribution(filtered_df)
                            st.plotly_chart(category_fig, use_container_width=True)
                    
                    # Tab 2: Source Comparison
                    with tabs[1]:
                        if 'source' in filtered_df.columns and len(filtered_df['source'].unique()) > 1:
                            sources = filtered_df['source'].unique()
                            source_fig = create_source_comparison(
                                filtered_df, 
                                sources,
                                chatbot.conversation_memory['current_context'].get('time_range')
                            )
                            st.plotly_chart(source_fig, use_container_width=True)
                        else:
                            st.info("Source comparison requires data from multiple sources.")
                    
                    # Tab 3: Seasonal Patterns
                    with tabs[2]:
                        if st.session_state.show_seasonal and len(filtered_df) >= 30:
                            weekday_fig, hourly_fig, monthly_fig = create_seasonal_analysis(filtered_df)
                            
                            # Display seasonal patterns in columns
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.plotly_chart(weekday_fig, use_container_width=True)
                                st.plotly_chart(monthly_fig, use_container_width=True)
                            
                            with col2:
                                st.plotly_chart(hourly_fig, use_container_width=True)
                                
                                # Show statistical insights
                                st.markdown("### Statistical Insights")
                                
                                # Calculate busy times
                                busy_hour = hourly_fig.data[0].y.argmax()
                                busy_day = weekday_fig.data[0].y.argmax()
                                busy_month = monthly_fig.data[0].y.argmax()
                                
                                st.markdown(f"- Busiest hour: **{busy_hour:02d}:00**")
                                st.markdown(f"- Busiest day: **{['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][busy_day]}**")
                                st.markdown(f"- Busiest month: **{calendar.month_name[busy_month+1]}**")
                        else:
                            st.info("Seasonal analysis requires more data points.")
            
            # Show sample messages
            if chatbot.conversation_memory['last_filtered_df'] is not None and len(chatbot.conversation_memory['last_filtered_df']) > 0:
                with st.expander("Sample Messages"):
                    sample_df = chatbot.conversation_memory['last_filtered_df'].sample(
                        min(10, len(chatbot.conversation_memory['last_filtered_df']))
                    )
                    if 'message' in sample_df.columns:
                        display_df = sample_df[['timestamp', 'message', 'category', 'source']].copy()
                        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                        display_df['category'] = display_df['category'].apply(
                            lambda x: chatbot.category_display_names.get(x, x.replace('_', ' ').title())
                        )
                        st.dataframe(display_df, use_container_width=True)
            
            # User feedback section
            with st.expander("Suggested Follow-up Questions"):
                # Generate follow-up questions based on the current context
                follow_ups = []
                
                # Compare with another source
                if 'source' in chatbot.conversation_memory['current_context'] and chatbot.conversation_memory['current_context']['source']:
                    current_source = chatbot.conversation_memory['current_context']['source']
                    other_source = 'telegram' if current_source == 'livechat' else 'livechat'
                    follow_ups.append(f"What about {other_source}?")
                
                # Time-based follow-ups
                if 'time_range' in chatbot.conversation_memory['current_context'] and chatbot.conversation_memory['current_context']['time_range']:
                    follow_ups.append("Show me trends over time")
                    follow_ups.append("Compare with the previous period")
                
                # Category-based follow-ups
                if 'category' in chatbot.conversation_memory['current_context'] and chatbot.conversation_memory['current_context']['category']:
                    current_category = chatbot.conversation_memory['current_context']['category']
                    follow_ups.append(f"Show me daily patterns for {current_category}")
                    
                    # Suggest related categories
                    related_categories = {
                        'account_access': ['payment_issues', 'technical_errors'],
                        'payment_issues': ['account_access', 'bonus_issues_general'],
                        'game_problems': ['technical_errors', 'freespins_issues'],
                        'bonus_issues_general': ['bonus_activation', 'bonus_missing'],
                        'freespins_issues': ['game_problems', 'bonus_issues_general']
                    }
                    
                    if current_category in related_categories:
                        related = related_categories[current_category][0]
                        related_name = chatbot.category_display_names.get(related, related.replace('_', ' ').title())
                        follow_ups.append(f"Compare with {related_name}")
                
                # Display suggestions as buttons
                if follow_ups:
                    col1, col2 = st.columns(2)
                    for i, question in enumerate(follow_ups):
                        if i % 2 == 0:
                            with col1:
                                if st.button(question, key=f"suggestion_{i}"):
                                    st.session_state.query = question
                                    st.rerun()
                        else:
                            with col2:
                                if st.button(question, key=f"suggestion_{i}"):
                                    st.session_state.query = question
                                    st.rerun()

        except Exception as e:
            st.error(f"Oops! Something went wrong: {e}")
            if st.checkbox("Show error details"):
                st.exception(e)
    elif query and not chatbot:
        st.error("Chatbot could not be loaded. Please check the dataset path.")
    else:
        # Welcome screen when no query is entered
        st.markdown("""
        <div class="stats-box">
            <h3>👋 Welcome to PETe Bot!</h3>
            <p>I can help you analyze user feedback messages across multiple categories and sources.</p>
            <p>Type a question in the box above to get started, or try one of the example queries from the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Preview of data and capabilities
        if chatbot and hasattr(chatbot, 'df'):
            # Show preview of available data
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Messages", len(chatbot.df))
            with col2:
                st.metric("Categories", len(chatbot.categories))
            with col3:
                st.metric("Data Sources", len(chatbot.sources))
            
            # Show date range of available data
            min_date = chatbot.df['timestamp'].min().strftime('%Y-%m-%d')
            max_date = chatbot.df['timestamp'].max().strftime('%Y-%m-%d')
            
            st.info(f"Data available from {min_date} to {max_date}")
            
            # Show top categories
            st.subheader("Available Categories")
            top_cats = chatbot.df['category'].value_counts().head(8)
            cat_names = [chatbot.category_display_names.get(cat, cat.replace('_', ' ').title()) 
                        for cat in top_cats.index]
            
            cat_fig = px.bar(
                x=cat_names,
                y=top_cats.values,
                labels={'x': 'Category', 'y': 'Number of Messages'},
                title="Top Message Categories"
            )
            st.plotly_chart(cat_fig, use_container_width=True)

if __name__ == "__main__":
    main()
