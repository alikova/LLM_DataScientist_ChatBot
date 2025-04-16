# Replace the time chart section with this code:
if 'charts' in response['data'] and response['data']['charts']:
    st.subheader("Time Analysis")
    for chart_data in response['data']['charts']:
        if chart_data['type'] == 'time_series' and hasattr(chart_data['data'], 'index'):
            # Get the time series data
            time_series = chart_data['data']
            
            # Ensure we have data spanning more time
            if len(time_series) > 1:
                # Calculate date range
                date_range = (time_series.index.max() - time_series.index.min()).days
                
                # For short date ranges, keep original daily view
                if date_range < 60:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(time_series.index, time_series.values, marker='o')
                    
                # For longer ranges, resample to show full year better
                else:
                    # Create monthly view for ranges over 60 days
                    monthly_data = time_series.resample('M').sum()
                    
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(monthly_data.index, monthly_data.values, marker='o', linewidth=2)
                    ax.set_title(f"Monthly Messages Over Time{' - ' + chart_data.get('category', '') if 'category' in chart_data else ''}")
                    
                    # Format x-axis to show month names
                    import matplotlib.dates as mdates
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                    
                    # Add year markers
                    years = sorted(list(set(d.year for d in monthly_data.index)))
                    for year in years:
                        plt.axvline(pd.Timestamp(f"{year}-01-01"), color='gray', linestyle='--', alpha=0.5)
                        plt.text(pd.Timestamp(f"{year}-01-01"), monthly_data.max() * 1.05, str(year), 
                                ha='center', va='bottom', fontsize=10)
                    
                    # Calculate and display year-over-year stats if we have multiple years
                    if len(years) > 1:
                        yearly_data = time_series.resample('Y').sum()
                        if len(yearly_data) > 1:
                            yoy_change = ((yearly_data.iloc[-1] - yearly_data.iloc[-2]) / yearly_data.iloc[-2]) * 100
                            change_text = f"Year-over-year change: {yoy_change:.1f}%"
                            plt.figtext(0.02, 0.02, change_text, fontsize=10, color='blue')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

# Add this code after displaying the time charts to show multi-category stats
# (Note that it seems your chatbot already has multi-category support)
st.subheader("Multi-Category Analysis")

# Check if the response already has multi-category data
if 'multi_category' in response['data'] and response['data']['multi_category']:
    # Get the multi-category results
    multi_cat_results = response['data']['category_results']
    
    # Display simple stats
    total_messages = response['data']['count']
    total_categories = len(multi_cat_results)
    
    # Create metrics
    st.markdown(f"**Messages span across {total_categories} categories**")
    
    # Show category distribution
    category_counts = [(result['category_name'], result['count'], 
                       (result['count']/total_messages)*100) 
                      for result in multi_cat_results]
    
    # Sort by count
    category_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Display as a simple table
    st.markdown("### Category Distribution")
    cols = st.columns(3)
    cols[0].markdown("**Category**")
    cols[1].markdown("**Count**")
    cols[2].markdown("**Percentage**")
    
    for cat_name, count, percent in category_counts:
        cols[0].markdown(f"{cat_name}")
        cols[1].markdown(f"{count}")
        cols[2].markdown(f"{percent:.1f}%")
    
    # Calculate category overlap percentage
    if len(multi_cat_results) > 1:
        # Calculate how many messages appear in multiple categories
        total_categorized = sum(result['count'] for result in multi_cat_results)
        overlap_estimate = total_categorized - total_messages
        overlap_percent = (overlap_estimate / total_messages) * 100
        
        if overlap_estimate > 0:
            st.info(f"Approximately {overlap_percent:.1f}% of messages could fit in multiple categories")

# If no explicit multi-category data, try to analyze the filtered dataframe
elif chatbot.conversation_memory['last_filtered_df'] is not None:
    filtered_df = chatbot.conversation_memory['last_filtered_df']
    
    if len(filtered_df) >= 10 and 'category' in filtered_df.columns:
        # Sample messages for quick analysis
        sample_size = min(100, len(filtered_df))
        sample_df = filtered_df.sample(sample_size)
        
        # Count unique primary categories
        unique_cats = sample_df['category'].nunique()
        
        # Simple message to indicate category distribution
        st.markdown(f"The analyzed messages span **{unique_cats}** distinct categories")
        
        # Display category distribution as percentages
        cat_counts = sample_df['category'].value_counts()
        for cat, count in cat_counts.items():
            percent = (count / len(sample_df)) * 100
            cat_name = chatbot.category_display_names.get(cat, cat.replace('_', ' ').title())
            st.markdown(f"- **{cat_name}**: {percent:.1f}% ({count} messages)")
        
        # Add a note about possible overlaps
        st.info("Note: Some messages may fit into multiple categories based on their content.")
