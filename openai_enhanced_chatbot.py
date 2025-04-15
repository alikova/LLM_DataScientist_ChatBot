"""
Enhanced Feedback Analysis Chatbot with OpenAI integration and conversational memory
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import Counter
from scipy import stats
try:
    import openai
except ImportError:
    print("OpenAI package not installed. Some features may not work.")

class EnhancedFeedbackChatbot:
    def __init__(self, data_path, openai_api_key=None):
        """
        Initialize the chatbot with data and optional OpenAI API key
        """
        # Load the data
        self.df = pd.read_csv(data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        # Get available categories and sources
        self.categories = sorted(self.df['category'].unique().tolist())
        self.sources = sorted(self.df['source'].unique().tolist())

        # Set up OpenAI if API key is provided
        self.use_openai = False
        if openai_api_key:
            try:
                openai.api_key = openai_api_key
                self.use_openai = True
            except:
                print("Failed to initialize OpenAI API. Continuing without it.")

        # Enhanced conversational memory with deeper history
        self.conversation_memory = {
            'messages': [],  # Full conversation history
            'last_query': None,
            'filter_history': [],  # History of filters, newest first
            'context_stack': [],   # Stack of contexts for navigation
            'current_context': {
                'category': None,
                'source': None,
                'time_range': None,
                'context_id': 0,   # Unique identifier for this context
                'query_intent': None  # Store the intent of the query
            },
            'last_filtered_df': None,
            'context_counter': 0   # Counter for assigning unique IDs to contexts
        }

        # For compatibility with original code
        self.context = self.conversation_memory['current_context']

        # Category mappings for better user experience
        self.category_mappings = {
            'login': 'account_access',
            'login issues': 'account_access',
            'account': 'account_access',
            'account issues': 'account_access',
            'access': 'account_access',
            'access issues': 'account_access',
            'password': 'account_access',
            'login access': 'account_access',
            'verification': 'account_access',

            'payment': 'payment_issues',
            'payment issues': 'payment_issues',
            'deposit': 'payment_issues',
            'deposit issues': 'payment_issues',
            'withdraw': 'payment_issues',
            'withdrawal': 'payment_issues',
            'withdrawal issues': 'payment_issues',
            'money': 'payment_issues',
            'transaction': 'payment_issues',

            'game': 'game_problems',
            'game issues': 'game_problems',
            'game problems': 'game_problems',
            'games': 'game_problems',
            'slot': 'game_problems',
            'slots': 'game_problems',
            'play': 'game_problems',
            'playing': 'game_problems',
            'bet': 'game_problems',
            'betting': 'game_problems',

            'bonus': 'bonus_issues_general',
            'bonus issues': 'bonus_issues_general',
            'bonuses': 'bonus_issues_general',
            'promo': 'bonus_issues_general',
            'promotion': 'bonus_issues_general',
            'promotions': 'bonus_issues_general',
            'offer': 'bonus_issues_general',
            'offers': 'bonus_issues_general',

            'bonus activation': 'bonus_activation',
            'activate bonus': 'bonus_activation',
            'bonus code': 'bonus_activation',
            'promo code': 'bonus_activation',
            'redeem': 'bonus_activation',
            'redeem code': 'bonus_activation',

            'missing bonus': 'bonus_missing',
            'bonus missing': 'bonus_missing',
            'no bonus': 'bonus_missing',
            'didn\'t receive bonus': 'bonus_missing',
            'not credited': 'bonus_missing',

            'bonus eligibility': 'bonus_eligibility',
            'eligible': 'bonus_eligibility',
            'eligibility': 'bonus_eligibility',
            'qualify': 'bonus_eligibility',
            'qualification': 'bonus_eligibility',

            'free spins': 'freespins_issues',
            'freespins': 'freespins_issues',
            'free spin': 'freespins_issues',
            'freespin': 'freespins_issues',
            'free games': 'freespins_issues',

            'error': 'technical_errors',
            'errors': 'technical_errors',
            'technical': 'technical_errors',
            'technical issues': 'technical_errors',
            'technical problems': 'technical_errors',
            'bug': 'technical_errors',
            'bugs': 'technical_errors',
            'site': 'technical_errors',
            'website': 'technical_errors',
            'app': 'technical_errors'
        }

        # Source mappings
        self.source_mappings = {
            'telegram': 'telegram',
            'livechat': 'livechat',
            'live chat': 'livechat'
        }

        # Pretty names for display
        self.category_display_names = {
            'account_access': 'Account Access Issues',
            'payment_issues': 'Payment Issues',
            'game_problems': 'Game Problems',
            'bonus_issues_general': 'General Bonus Issues',
            'bonus_activation': 'Bonus Activation Issues',
            'bonus_missing': 'Missing Bonus Issues',
            'bonus_eligibility': 'Bonus Eligibility Issues',
            'freespins_issues': 'Free Spins Issues',
            'technical_errors': 'Technical Errors',
            'general_inquiry': 'General Inquiries'
        }

        print(f"Enhanced chatbot initialized with {len(self.df)} messages")
        print(f"OpenAI API integration: {'Enabled' if self.use_openai else 'Disabled'}")

    def _extract_filters_with_openai(self, query):
        """Use OpenAI to extract filters from query"""
        try:
            if not self.use_openai:
                raise ValueError("OpenAI API not configured")

            system_prompt = """
            You are a helpful assistant that extracts filtering parameters from user queries about feedback data.
            Extract the following information:
            1. Category (if mentioned): One of [account_access, payment_issues, game_problems, bonus_issues_general, bonus_activation, bonus_missing, bonus_eligibility, freespins_issues, technical_errors]
            2. Source (if mentioned): One of [telegram, livechat]
            3. Time range (if mentioned): Express as a tuple (start_date, end_date) in ISO format

            Return only a JSON object with these three fields. If a field is not mentioned, set it to null.
            """

            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversation,
                temperature=0.1
            )

            result = response.choices[0].message.content
            import json
            extracted = json.loads(result)

            # Process time range if present
            if extracted.get('time_range'):
                start_str, end_str = extracted['time_range']
                extracted['time_range'] = (
                    datetime.fromisoformat(start_str.replace('Z', '+00:00')),
                    datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                )

            return extracted

        except Exception as e:
            print(f"Error using OpenAI to extract filters: {e}")
            # Fall back to regular methods
            return {
                'category': self._parse_category(query),
                'source': self._parse_source(query),
                'time_range': self._parse_time_range(query)
            }

    def _parse_time_range(self, query):
        """
        Extract time range from the query text
        Now with support for specific month-year combinations

        Args:
            query (str): The user's query text

        Returns:
            tuple or None: (start_time, end_time) if found, None otherwise
        """
        query = query.lower()

        # Month name mapping
        month_names = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }

        # Define time range patterns and their corresponding functions
        time_patterns = [
            # Month with year (e.g., "January 2024" or "Jan 2024")
            (r'(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|sept|october|oct|november|nov|december|dec)\s+(\d{4})',
            lambda x: (
                datetime(int(x.group(2)), month_names[x.group(1)], 1),
                datetime(int(x.group(2)), month_names[x.group(1)],
                          self._get_last_day_of_month(int(x.group(2)), month_names[x.group(1)]))
            )),

            # Year only (e.g., "2024")
            (r'\b(\d{4})\b',
            lambda x: (
                datetime(int(x.group(1)), 1, 1),
                datetime(int(x.group(1)), 12, 31)
            )),

            # Quarter with year (e.g., "Q1 2024" or "first quarter 2024")
            (r'(?:q(\d)|(?:first|1st|second|2nd|third|3rd|fourth|4th) quarter)\s+(\d{4})',
            lambda x: self._parse_quarter(x)),

            # Specific number of days/weeks/months
            (r'last (\d+) days?', lambda x: (datetime.now() - timedelta(days=int(x.group(1))), datetime.now())),
            (r'last (\d+) weeks?', lambda x: (datetime.now() - timedelta(weeks=int(x.group(1))), datetime.now())),
            (r'last (\d+) months?', lambda x: (datetime.now() - timedelta(days=int(x.group(1))*30), datetime.now())),

            # Common time expressions
            (r'yesterday', lambda x: (datetime.now() - timedelta(days=1), datetime.now())),
            (r'last day', lambda x: (datetime.now() - timedelta(days=1), datetime.now())),
            (r'last week', lambda x: (datetime.now() - timedelta(weeks=1), datetime.now())),
            (r'last month', lambda x: (datetime.now() - timedelta(days=30), datetime.now())),
            (r'last year', lambda x: (datetime.now() - timedelta(days=365), datetime.now())),
            (r'this week', lambda x: (datetime.now() - timedelta(days=datetime.now().weekday()), datetime.now())),
            (r'this month', lambda x: (datetime.now().replace(day=1), datetime.now())),
            (r'today', lambda x: (datetime.now().replace(hour=0, minute=0, second=0), datetime.now())),

            # Date ranges
            (r'between ([\d-]+) and ([\d-]+)', lambda x: (
                datetime.strptime(x.group(1), '%Y-%m-%d'),
                datetime.strptime(x.group(2), '%Y-%m-%d')
            )),
            (r'from ([\d-]+) to ([\d-]+)', lambda x: (
                datetime.strptime(x.group(1), '%Y-%m-%d'),
                datetime.strptime(x.group(2), '%Y-%m-%d')
            )),

            # Since a specific date
            (r'since ([\d-]+)', lambda x: (
                datetime.strptime(x.group(1), '%Y-%m-%d'),
                datetime.now()
            ))
        ]

        # Try each pattern
        for pattern, time_func in time_patterns:
            match = re.search(pattern, query)
            if match:
                try:
                    return time_func(match)
                except Exception as e:
                    print(f"Error parsing time range: {e}")

        return None

    def _get_last_day_of_month(self, year, month):
        """
        Helper method to get the last day of a specific month

        Args:
            year (int): The year
            month (int): The month (1-12)

        Returns:
            int: The last day of the month
        """
        # If it's December, the next month is January of next year
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)

        # Last day is one day before the 1st of next month
        return (next_month - timedelta(days=1)).day

    def _parse_quarter(self, match):
        """
        Helper method to parse quarter references

        Args:
            match: The regex match object

        Returns:
            tuple: (start_date, end_date) for the quarter
        """
        # Check if we have a numeric quarter (Q1, Q2, etc.) or a textual one (first quarter, etc.)
        if match.group(1) and match.group(1).isdigit():
            quarter = int(match.group(1))
            year = int(match.group(2))
        else:
            # Handle textual quarters
            quarter_text = match.group(1).lower()
            if 'first' in quarter_text or '1st' in quarter_text:
                quarter = 1
            elif 'second' in quarter_text or '2nd' in quarter_text:
                quarter = 2
            elif 'third' in quarter_text or '3rd' in quarter_text:
                quarter = 3
            elif 'fourth' in quarter_text or '4th' in quarter_text:
                quarter = 4
            year = int(match.group(2))

        # Calculate start and end dates based on quarter
        start_month = (quarter - 1) * 3 + 1
        end_month = quarter * 3

        start_date = datetime(year, start_month, 1)

        # Calculate the last day of the end month
        if end_month == 12:
            end_date = datetime(year, 12, 31)
        else:
            next_month = datetime(year, end_month + 1, 1)
            end_date = next_month - timedelta(days=1)

        return (start_date, end_date)

    def _parse_category(self, query):
        """
        Extract category from the query text

        Args:
            query (str): The user's query text

        Returns:
            str or None: Category name if found, None otherwise
        """
        query = query.lower()

        # Try direct category matching
        for key, category in self.category_mappings.items():
            if key in query:
                return category

        return None

    def _parse_source(self, query):
        """
        Extract source from the query text

        Args:
            query (str): The user's query text

        Returns:
            str or None: Source name if found, None otherwise
        """
        query = query.lower()

        # Try direct source matching
        for key, source in self.source_mappings.items():
            if key in query:
                return source

        return None
    
    def process_query(self, query):
        """
        Process a query with support for multiple categories
        """
        # For queries that are likely asking for specific category comparisons
        if any(phrase in query.lower() for phrase in ['compare', 'versus', 'vs', 'difference between']):
            return self.process_comparison_query(query)
        
        # For normal queries that might involve multiple categories
        return self.process_query_with_multiple_categories(query)

    def _is_follow_up_query(self, query):
        """
        Check if the query is a follow-up to a previous question

        Args:
            query (str): The user's query text

        Returns:
            bool: True if it's a follow-up query, False otherwise
        """
        query = query.lower()
        follow_up_phrases = [
            'what about', 'how about', 'show me', 'and what about',
            'what if', 'compare with', 'compare to', 'versus',
            'vs', 'instead', 'rather', 'similar', 'same for'
        ]

        return any(phrase in query for phrase in follow_up_phrases)

    def process_query_with_multiple_categories(self, query):
        """
        Process a query that may involve multiple categories
        
        Args:
            query (str): The user's query text
            
        Returns:
            dict: Response with message and data
        """
        # Add query to conversation history
        self.conversation_memory['messages'].append({
            'role': 'user',
            'content': query
        })
        self.conversation_memory['last_query'] = query
        
        # Extract source and time range as usual
        source = self._parse_source(query)
        time_range = self._parse_time_range(query)
        
        # Extract multiple categories with confidence
        sample_df = self.df.sample(min(500, len(self.df)))
        categories_with_confidence = self._extract_categories_with_confidence(query, sample_df)
        
        # Create extracted_filters dict for use with existing methods
        extracted_filters = {
            'category': categories_with_confidence[0][0] if categories_with_confidence else None,
            'source': source,
            'time_range': time_range
        }
        
        # Determine query intent
        query_intent = self._determine_query_intent(query, extracted_filters)
        
        # Update current context based on query intent
        if query_intent == 'new_topic':
            new_context = {
                'category': categories_with_confidence[0][0] if categories_with_confidence else None,
                'multi_categories': categories_with_confidence,
                'source': source,
                'time_range': time_range,
                'context_id': self.conversation_memory['context_counter'],
                'query_intent': 'new_topic'
            }
            self.conversation_memory['context_counter'] += 1

            # Save current context to history before replacing it
            if any(value is not None for key, value in self.conversation_memory['current_context'].items()
                  if key not in ['context_id', 'query_intent']):
                self.conversation_memory['filter_history'].insert(
                    0, self.conversation_memory['current_context'].copy()
                )
                # Keep the history to a reasonable size
                if len(self.conversation_memory['filter_history']) > 10:
                    self.conversation_memory['filter_history'].pop()

            # Update current context
            self.conversation_memory['current_context'] = new_context

            # Add to context stack for navigation
            self.conversation_memory['context_stack'].append(new_context.copy())
            if len(self.conversation_memory['context_stack']) > 15:
                self.conversation_memory['context_stack'].pop(0)

        elif query_intent == 'follow_up':
            # Follow-up to the current context - modify existing context
            current_context = self.conversation_memory['current_context'].copy()

            # Update only specified filters, keep the rest
            for key, value in extracted_filters.items():
                if value is not None:
                    current_context[key] = value
            
            # Update multi-categories if we have new ones
            if categories_with_confidence:
                current_context['multi_categories'] = categories_with_confidence

            current_context['query_intent'] = 'follow_up'

            # Save to history
            self.conversation_memory['filter_history'].insert(0, self.conversation_memory['current_context'].copy())
            if len(self.conversation_memory['filter_history']) > 10:
                self.conversation_memory['filter_history'].pop()

            # Update current
            self.conversation_memory['current_context'] = current_context

        elif query_intent == 'context_switch':
            # User wants to switch to a previous context
            target_context = self._find_target_context(query)
            if target_context:
                # Save current to history before switching
                self.conversation_memory['filter_history'].insert(
                    0, self.conversation_memory['current_context'].copy()
                )
                if len(self.conversation_memory['filter_history']) > 10:
                    self.conversation_memory['filter_history'].pop()

                # Set as current
                self.conversation_memory['current_context'] = target_context.copy()
                self.conversation_memory['current_context']['query_intent'] = 'context_switch'
            else:
                # If no specific context found, treat as follow-up
                current_context = self.conversation_memory['current_context'].copy()

                # Update specified filters
                for key, value in extracted_filters.items():
                    if value is not None:
                        current_context[key] = value
                
                # Update multi-categories if we have new ones
                if categories_with_confidence:
                    current_context['multi_categories'] = categories_with_confidence

                current_context['query_intent'] = 'follow_up'

                # Save to history and update current
                self.conversation_memory['filter_history'].insert(0, self.conversation_memory['current_context'].copy())
                self.conversation_memory['current_context'] = current_context

        elif query_intent == 'comparison':
            # Comparison between current and another context
            # Handle specially - we might need to process two contexts
            comparison_context = self._extract_comparison_context(query)

            # Process the comparison...
            current_context = self.conversation_memory['current_context'].copy()

            # Update specified filters
            for key, value in extracted_filters.items():
                if value is not None:
                    current_context[key] = value
            
            # Update multi-categories if we have new ones
            if categories_with_confidence:
                current_context['multi_categories'] = categories_with_confidence

            current_context['query_intent'] = 'comparison'
            current_context['comparison_target'] = comparison_context

            # Save and update
            self.conversation_memory['filter_history'].insert(0, self.conversation_memory['current_context'].copy())
            self.conversation_memory['current_context'] = current_context

        # For multi-category processing, we'll create a composite result
        if len(categories_with_confidence) > 1:
            # Initialize base dataframe with all filters except category
            filtered_base_df = self.df.copy()
            
            # Apply source filter
            if source:
                filtered_base_df = filtered_base_df[filtered_base_df['source'] == source]
            
            # Apply time range filter
            if time_range:
                start_time, end_time = time_range
                filtered_base_df = filtered_base_df[
                    (filtered_base_df['timestamp'] >= start_time) &
                    (filtered_base_df['timestamp'] <= end_time)
                ]
            
            # Prepare a combined response
            multi_category_results = []
            primary_category_df = None
            
            # Process each category with sufficient confidence
            for category, confidence in categories_with_confidence:
                if confidence >= 40:  # Only include categories with sufficient confidence
                    # Filter by this category
                    category_df = filtered_base_df[filtered_base_df['category'] == category]
                    
                    if len(category_df) > 0:
                        # Store primary category DataFrame (highest confidence)
                        if primary_category_df is None:
                            primary_category_df = category_df
                        
                        # Get stats for this category
                        message_count = len(category_df)
                        unique_users = category_df['id_user'].nunique()
                        
                        # Category display name
                        category_name = self.category_display_names.get(
                            category, category.replace('_', ' ').title()
                        )
                        
                        # Add to results
                        multi_category_results.append({
                            'category': category,
                            'category_name': category_name,
                            'confidence': confidence,
                            'count': message_count,
                            'unique_users': unique_users
                        })
            
            # Generate combined response
            if multi_category_results:
                # Sort by count (more relevant results first)
                multi_category_results.sort(key=lambda x: x['count'], reverse=True)
                
                # Generate message
                message_parts = []
                
                # Add context switch message if applicable
                if query_intent == 'context_switch':
                    context_desc = self._get_context_description(self.conversation_memory['current_context'])
                    message_parts.append(f"I've switched to the context about {context_desc}.")
                
                # Intro sentence
                if source or time_range:
                    filter_desc = []
                    if source:
                        filter_desc.append(f"from {source.title()}")
                    if time_range:
                        start_time, end_time = time_range
                        time_desc = f"between {start_time.strftime('%Y-%m-%d')} and {end_time.strftime('%Y-%m-%d')}"
                        filter_desc.append(time_desc)
                    
                    message_parts.append(f"I found messages {' '.join(filter_desc)} across multiple categories:")
                else:
                    message_parts.append("I found messages across multiple categories:")
                
                # Add category breakdowns
                for result in multi_category_results:
                    message_parts.append(
                        f"- {result['category_name']}: {result['count']} messages from {result['unique_users']} users "
                        f"(confidence: {result['confidence']:.0f}%)"
                    )
                
                # Set primary filtered DataFrame for additional analysis
                self.conversation_memory['last_filtered_df'] = primary_category_df
                
                # Create charts based on the primary category
                charts = []
                if primary_category_df is not None and len(primary_category_df) >= 5:
                    # Create time series chart
                    primary_category_df['date'] = primary_category_df['timestamp'].dt.date
                    daily_counts = primary_category_df.groupby('date').size()
                    
                    # Check for spikes
                    if len(daily_counts) > 3:
                        mean = daily_counts.mean()
                        std = daily_counts.std()
                        threshold = mean + 1.5 * std

                        spikes = daily_counts[daily_counts > threshold]
                        if not spikes.empty:
                            if len(spikes) == 1:
                                spike_date = spikes.index[0]
                                message_parts.append(f"I noticed a significant spike in {multi_category_results[0]['category_name']} on {spike_date}.")
                            else:
                                spike_dates = ", ".join([str(date) for date in spikes.index])
                                message_parts.append(f"I noticed significant spikes in {multi_category_results[0]['category_name']} on these dates: {spike_dates}.")
                    
                    # Add the time series chart
                    charts.append({
                        'type': 'time_series',
                        'data': daily_counts,
                        'category': multi_category_results[0]['category_name']
                    })
                    
                    # Add enhanced temporal analysis if we have enough data
                    temporal_insights = []
                    
                    # Check for seasonal patterns
                    if len(primary_category_df) >= 30:  # At least 30 data points
                        seasonal_patterns = self.detect_seasonal_patterns(primary_category_df)
                        temporal_insights.extend(seasonal_patterns)
                    
                    # Check for trends
                    if len(primary_category_df) >= 21:  # Enough for a 7-day window
                        trend_insights = self.detect_trends(primary_category_df, window=7)
                        temporal_insights.extend(trend_insights)
                    
                    # Year-over-year comparison if sufficient data
                    if primary_category_df['timestamp'].max() - primary_category_df['timestamp'].min() >= pd.Timedelta(days=365):
                        yoy_insights = self.compare_year_over_year(primary_category_df, multi_category_results[0]['category'])
                        temporal_insights.extend(yoy_insights)
                    
                    # Add temporal insights to message
                    if temporal_insights:
                        message_parts.append(f"\nI noticed these patterns over time for {multi_category_results[0]['category_name']}:")
                        for insight in temporal_insights:
                            message_parts.append(f"- {insight}")
                
                # Add suggestions for exploring categories
                category_suggestions = []
                if len(multi_category_results) > 1:
                    message_parts.append("\nYou might want to explore these categories in more detail:")
                    for result in multi_category_results[:2]:  # Top 2 categories
                        suggestion = f"Tell me more about {result['category_name']}"
                        if source:
                            suggestion += f" from {source}"
                        if time_range:
                            start_time, end_time = time_range
                            suggestion += f" between {start_time.strftime('%Y-%m-%d')} and {end_time.strftime('%Y-%m-%d')}"
                        category_suggestions.append(suggestion)
                        message_parts.append(f"- {suggestion}")
                
                return {
                    'message': "\n".join(message_parts),
                    'data': {
                        'count': sum(result['count'] for result in multi_category_results),
                        'unique_users': len(set().union(*(
                            set(filtered_base_df[filtered_base_df['category'] == result['category']]['id_user'].unique())
                            for result in multi_category_results
                        ))),
                        'charts': charts,
                        'multi_category': True,
                        'category_results': multi_category_results,
                        'category_suggestions': category_suggestions
                    }
                }
        
        # If we reach here, it means we have a single category or no categories
        # Apply filters to the data based on current context
        filtered_df = self.df.copy()

        # Apply category filter
        if self.conversation_memory['current_context']['category']:
            filtered_df = filtered_df[filtered_df['category'] == self.conversation_memory['current_context']['category']]

        # Apply source filter
        if self.conversation_memory['current_context']['source']:
            filtered_df = filtered_df[filtered_df['source'] == self.conversation_memory['current_context']['source']]

        # Apply time range filter
        if self.conversation_memory['current_context']['time_range']:
            start_time, end_time = self.conversation_memory['current_context']['time_range']
            filtered_df = filtered_df[
                (filtered_df['timestamp'] >= start_time) &
                (filtered_df['timestamp'] <= end_time)
            ]

        # Store the filtered dataframe
        self.conversation_memory['last_filtered_df'] = filtered_df

        # Generate response
        response = self._generate_response(
            filtered_df,
            self.conversation_memory['current_context']['category'],
            self.conversation_memory['current_context']['source'],
            self.conversation_memory['current_context']['time_range'],
            query_intent
        )

        # Add response to conversation history
        self.conversation_memory['messages'].append({
            'role': 'assistant',
            'content': response['message']
        })

        return response

    def suggest_category_exploration(self, query_results):
        """
        Generate suggestions for exploring related categories
        
        Args:
            query_results (dict): The results from processing a query
            
        Returns:
            list: Suggested follow-up queries for category exploration
        """
        suggestions = []
        
        # If we have multiple category results
        if query_results.get('multi_category') and 'category_results' in query_results:
            category_results = query_results['category_results']
            
            # Skip if only one category
            if len(category_results) <= 1:
                return suggestions
            
            # Get time and source from current context
            current_context = self.conversation_memory['current_context']
            time_filter = ""
            if current_context.get('time_range'):
                start, end = current_context['time_range']
                time_filter = f" between {start.strftime('%Y-%m-%d')} and {end.strftime('%Y-%m-%d')}"
            
            source_filter = ""
            if current_context.get('source'):
                source_filter = f" from {current_context['source'].title()}"
            
            # Suggest exploring specific categories
            for result in category_results[:3]:  # Top 3 categories
                category_name = result['category_name']
                suggestion = f"Tell me more about {category_name}{source_filter}{time_filter}"
                suggestions.append(suggestion)
            
            # Suggest comparing categories
            if len(category_results) >= 2:
                cat1 = category_results[0]['category_name']
                cat2 = category_results[1]['category_name']
                comparison = f"Compare {cat1} and {cat2}{source_filter}{time_filter}"
                suggestions.append(comparison)
            
            # Suggest time trends for main category
            main_category = category_results[0]['category_name']
            suggestions.append(f"Show me trends over time for {main_category}{source_filter}")
        
        return suggestions
    

    def _determine_query_intent(self, query, extracted_filters):
        """
        Determine the intent of the query to guide processing

        Types of intents:
        - new_topic: A completely new query
        - follow_up: A follow-up to the current context
        - context_switch: Request to switch to a previous context
        - comparison: Comparison between contexts
        """
        query_lower = query.lower()

        # Check if query is empty - shouldn't happen but let's be safe
        if not query or not query.strip():
            return 'follow_up'  # Default to follow-up

        # Check for comparison intent
        comparison_phrases = [
            'compare', 'versus', 'vs', 'difference', 'different',
            'compare to', 'compared to', 'comparison', 'contrast'
        ]
        if any(phrase in query_lower for phrase in comparison_phrases):
            return 'comparison'

        # Check for context switch intent
        switch_phrases = [
            'go back to', 'return to', 'switch to', 'earlier', 'previous',
            'last time', 'before', 'first question', 'you mentioned'
        ]
        if any(phrase in query_lower for phrase in switch_phrases):
            return 'context_switch'

        # Check for follow-up
        follow_up_phrases = [
            'what about', 'how about', 'and what', 'show me', 'can you tell me',
            'and', 'also', 'as well', 'too', 'additionally'
        ]

        # If it's short and doesn't have much specificity, likely a follow-up
        if len(query.split()) < 5 and all(value is None for value in extracted_filters.values()):
            return 'follow_up'

        if any(phrase in query_lower for phrase in follow_up_phrases):
            return 'follow_up'

        # Check if any filter is specified - if so, could be a new topic
        if any(value is not None for value in extracted_filters.values()):
            # If completely different from current context, it's a new topic
            current_context = self.conversation_memory['current_context']

            # Check if any filter matches the current context
            matching_filters = 0
            for key, value in extracted_filters.items():
                if value is not None and value == current_context.get(key):
                    matching_filters += 1

            # If matches current context partially, it's likely a follow-up rather than new topic
            if matching_filters > 0:
                return 'follow_up'

            # Otherwise it's a new topic
            return 'new_topic'

        # Default to follow-up for simpler handling
        return 'follow_up'

    def _find_target_context(self, query):
        """
        Find the target context when the user wants to switch to a previous context
        """
        query_lower = query.lower()

        # Check for numeric references like "2 queries ago"
        numeric_match = re.search(r'(\d+)\s*(quer|question|context)s?\s*ago', query_lower)
        if numeric_match:
            index = int(numeric_match.group(1))
            if 0 < index < len(self.conversation_memory['filter_history']):
                return self.conversation_memory['filter_history'][index]

        # Check for "first" or "original" context
        if 'first' in query_lower or 'original' in query_lower:
            if len(self.conversation_memory['filter_history']) > 0:
                return self.conversation_memory['filter_history'][-1]

        # Check for "previous" or "last" context
        if 'previous' in query_lower or 'last' in query_lower:
            if len(self.conversation_memory['filter_history']) > 0:
                return self.conversation_memory['filter_history'][0]

        # Try to match based on content (e.g., "go back to the payment issues")
        for context in self.conversation_memory['filter_history']:
            # Check if category matches query
            if context['category'] and self._category_mentioned_in_query(context['category'], query_lower):
                return context

            # Check if source matches query
            if context['source'] and context['source'].lower() in query_lower:
                return context

        # If no specific match, return the most recent different context
        current_cat = self.conversation_memory['current_context'].get('category')
        current_src = self.conversation_memory['current_context'].get('source')

        for context in self.conversation_memory['filter_history']:
            if context.get('category') != current_cat or context.get('source') != current_src:
                return context

        # If no suitable context found
        return None

    def _category_mentioned_in_query(self, category, query):
        """Check if a category is mentioned in the query"""
        # Direct check
        if category in query:
            return True

        # Check for display name
        display_name = self.category_display_names.get(category, '').lower()
        if display_name and display_name in query:
            return True

        # Check aliases in category mappings
        for key, value in self.category_mappings.items():
            if value == category and key in query:
                return True

        return False

    def _extract_comparison_context(self, query):
        """
        Extract the context to compare with when handling a comparison query
        Returns a context dict or None if can't determine
        """
        query_lower = query.lower()

        # Extract the comparison target
        comparison_targets = []

        # Check for source comparisons
        for source in self.sources:
            if source.lower() in query_lower:
                comparison_targets.append(('source', source))

        # Check for category comparisons
        for category_key, category_value in self.category_mappings.items():
            if category_key in query_lower:
                comparison_targets.append(('category', category_value))

        # If we found potential targets, create a comparison context
        if comparison_targets:
            comparison_context = {
                'category': None,
                'source': None,
                'time_range': self.conversation_memory['current_context'].get('time_range'),
                'context_id': -1,  # Special ID for comparison contexts
                'query_intent': 'comparison_target'
            }

            # Set the comparison attributes
            for attr_type, attr_value in comparison_targets:
                comparison_context[attr_type] = attr_value

            return comparison_context

        # If no specific comparison found, try to get the most recent different context
        current_cat = self.conversation_memory['current_context'].get('category')
        current_src = self.conversation_memory['current_context'].get('source')

        for context in self.conversation_memory['filter_history']:
            if context.get('category') != current_cat or context.get('source') != current_src:
                return context

        return None

    def _extract_categories_with_confidence(self, query, message_sample=None):
        """
        Extract multiple potential categories with confidence scores
        
        Args:
            query (str): The user's query text
            message_sample (DataFrame, optional): Sample of messages for context
            
        Returns:
            list: List of (category, confidence) tuples sorted by confidence
        """
        query_lower = query.lower()
        categories_scores = []
        
        # Method 1: Keyword matching with weighted scores
        for key, category in self.category_mappings.items():
            # Check for exact matches (full words)
            pattern = r'\b' + re.escape(key) + r'\b'
            matches = re.findall(pattern, query_lower)
            
            if matches:
                # Calculate base confidence based on key length and position
                # Longer keywords and those appearing earlier get higher confidence
                for match in matches:
                    position_factor = 1.0 - (query_lower.find(match) / len(query_lower)) * 0.5
                    length_factor = min(1.0, len(match) / 10)  # Cap at 1.0
                    confidence = (0.6 + (position_factor * 0.2) + (length_factor * 0.2)) * 100
                    categories_scores.append((category, confidence))
        
        # Method 2: If using OpenAI (more sophisticated approach)
        if self.use_openai and message_sample is not None and len(message_sample) > 0:
            try:
                # Create a prompt with message samples and the user query
                system_prompt = """
                You are an expert at categorizing user feedback for a gaming platform.
                Based on the sample messages and the user's query, rank the potential categories with confidence scores.
                Available categories:
                - account_access: Issues with login, passwords, verification
                - payment_issues: Problems with deposits, withdrawals, transactions
                - game_problems: Issues with gameplay, slots, betting
                - bonus_issues_general: General bonus and promotion issues
                - bonus_activation: Problems activating bonuses or using promo codes
                - bonus_missing: Missing or not credited bonuses
                - bonus_eligibility: Questions about bonus qualification
                - freespins_issues: Problems with free spins features
                - technical_errors: Website, app, or technical issues
                - general_inquiry: General questions and information requests
                
                Respond with a JSON object containing category names and confidence scores (0-100).
                Example: {"payment_issues": 85, "bonus_missing": 60}
                """
                
                # Format sample messages for context
                message_examples = "\n".join([
                    f"Category: {row['category']}, Message: {row['message']}"
                    for _, row in message_sample.sample(min(5, len(message_sample))).iterrows()
                ])
                
                user_prompt = f"""
                Sample messages from our system:
                {message_examples}
                
                User query to categorize: "{query}"
                
                Which categories apply to this query? Rank them with confidence scores.
                """
                
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=conversation,
                    temperature=0.1
                )
                
                result = response.choices[0].message.content
                
                # Parse JSON response
                import json
                ai_categories = json.loads(result)
                
                # Add AI-suggested categories to our list
                for category, confidence in ai_categories.items():
                    categories_scores.append((category, float(confidence)))
            
            except Exception as e:
                print(f"Error using OpenAI for multi-category detection: {e}")
        
        # Method 3: Use TF-IDF similarity if we don't have OpenAI
        if not self.use_openai and message_sample is not None and len(message_sample) > 0:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np
                
                # Group sample messages by category
                category_messages = {}
                for category in self.categories:
                    cat_msgs = message_sample[message_sample['category'] == category]['message'].tolist()
                    if cat_msgs:
                        category_messages[category] = ' '.join(cat_msgs)
                
                if category_messages:
                    # Create corpus with user query at the end
                    corpus = list(category_messages.values())
                    corpus.append(query)
                    
                    # Calculate TF-IDF and similarities
                    vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(corpus)
                    
                    # Get similarity between query and each category corpus
                    query_idx = len(corpus) - 1
                    similarities = cosine_similarity(tfidf_matrix[query_idx:query_idx+1], tfidf_matrix[:-1])[0]
                    
                    # Add categories with similarity scores
                    for idx, (category, _) in enumerate(category_messages.items()):
                        similarity = similarities[idx]
                        if similarity > 0.1:  # Minimum similarity threshold
                            confidence = similarity * 100
                            categories_scores.append((category, confidence))
            
            except Exception as e:
                print(f"Error using TF-IDF for multi-category detection: {e}")
        
        # Aggregate and normalize confidence scores
        aggregated_scores = {}
        for category, confidence in categories_scores:
            if category in aggregated_scores:
                # Take maximum confidence if found multiple times
                aggregated_scores[category] = max(aggregated_scores[category], confidence)
            else:
                aggregated_scores[category] = confidence
        
        # Sort by confidence (descending)
        sorted_categories = sorted(
            aggregated_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Filter to categories with minimum confidence
        filtered_categories = [(cat, score) for cat, score in sorted_categories if score >= 30]
        
        return filtered_categories

    

    def _generate_response(self, filtered_df, category, source, time_range, query_intent=None):
        """
        Enhanced generate_response that considers the query intent

        Args:
            filtered_df (DataFrame): The filtered data
            category (str): The category filter, if any
            source (str): The source filter, if any
            time_range (tuple): The time range filter (start_time, end_time), if any
            query_intent (str): The intent of the query (new_topic, follow_up, etc.)

        Returns:
            dict: Contains response message and visualization data
        """
        # Initialize message_parts with a prefix based on query intent
        message_parts = []

        # Handle context switch - add info about switching contexts
        if query_intent == 'context_switch':
            context_desc = self._get_context_description(self.conversation_memory['current_context'])
            message_parts.append(f"I've switched to the context about {context_desc}.")

        # Handle comparison queries
        if query_intent == 'comparison' and 'comparison_target' in self.conversation_memory['current_context']:
            comparison_target = self.conversation_memory['current_context']['comparison_target']
            if comparison_target:
                # Create a separate filtered df for the comparison context
                comparison_df = self.df.copy()

                # Apply filters from the comparison context
                if comparison_target.get('category'):
                    comparison_df = comparison_df[comparison_df['category'] == comparison_target['category']]

                if comparison_target.get('source'):
                    comparison_df = comparison_df[comparison_df['source'] == comparison_target['source']]

                if comparison_target.get('time_range'):
                    start_time, end_time = comparison_target['time_range']
                    comparison_df = comparison_df[
                        (comparison_df['timestamp'] >= start_time) &
                        (comparison_df['timestamp'] <= end_time)
                    ]

                # Get counts for comparison
                primary_count = len(filtered_df)
                comparison_count = len(comparison_df)

                # Describe what we're comparing
                primary_desc = self._get_context_description(self.conversation_memory['current_context'])
                comparison_desc = self._get_context_description(comparison_target)

                # Add comparison message
                message_parts.append(f"Comparing {primary_desc} ({primary_count} messages) with {comparison_desc} ({comparison_count} messages).")

                # Add percentage difference
                if comparison_count > 0:
                    percent_diff = ((primary_count - comparison_count) / comparison_count) * 100
                    if percent_diff > 0:
                        message_parts.append(f"That's {abs(percent_diff):.1f}% more messages than {comparison_desc}.")
                    else:
                        message_parts.append(f"That's {abs(percent_diff):.1f}% fewer messages than {comparison_desc}.")

        # If no messages match the filters
        if len(filtered_df) == 0:
            return {
                'message': "I couldn't find any messages matching your criteria.",
                'data': {
                    'count': 0,
                    'unique_users': 0,
                    'charts': []
                }
            }

        # Basic statistics
        message_count = len(filtered_df)
        unique_users = filtered_df['id_user'].nunique()

        # Create description of what was found
        filter_description = []

        if category:
            category_name = self.category_display_names.get(category, category.replace('_', ' ').title())
            filter_description.append(f"{category_name}")

        if source:
            filter_description.append(f"from {source.title()}")

        if time_range:
            start_time, end_time = time_range
            time_desc = f"between {start_time.strftime('%Y-%m-%d')} and {end_time.strftime('%Y-%m-%d')}"
            filter_description.append(time_desc)

        # Don't add the basic description if this is a context switch or comparison
        # (we've already added a more specific description)
        if query_intent not in ['context_switch', 'comparison']:
            if filter_description:
                filters_text = " ".join(filter_description)
                message_parts.append(f"I found {message_count} messages about {filters_text}.")
            else:
                message_parts.append(f"I found {message_count} messages in total.")

            message_parts.append(f"These messages came from {unique_users} unique users.")

        # Add source distribution if not filtered by source
        if not source and len(filtered_df['source'].unique()) > 1:
            source_counts = filtered_df['source'].value_counts()
            source_text = ", ".join([f"{count} from {src.title()}" for src, count in source_counts.items()])
            message_parts.append(f"Source breakdown: {source_text}.")

        # Add category distribution if not filtered by category
        if not category and len(filtered_df['category'].unique()) > 1:
            category_counts = filtered_df['category'].value_counts().head(3)
            category_text = ", ".join([
                f"{count} {self.category_display_names.get(cat, cat.replace('_', ' ').title())}"
                for cat, count in category_counts.items()
            ])
            message_parts.append(f"Top categories: {category_text}.")

        charts = []
        temporal_insights = []

        if len(filtered_df) >= 5:
            # [Existing code for basic time series analysis]
            
            # Check for seasonal patterns
            if len(filtered_df) >= 30:  # At least 30 data points
                seasonal_patterns = self.detect_seasonal_patterns(filtered_df)
                temporal_insights.extend(seasonal_patterns)
            
            # Check for trends
            if len(filtered_df) >= 21:  # Enough for a 7-day window
                trend_insights = self.detect_trends(filtered_df)
                temporal_insights.extend(trend_insights)
            
            # Year-over-year comparison if sufficient data
            if filtered_df['timestamp'].max() - filtered_df['timestamp'].min() >= pd.Timedelta(days=365):
                yoy_insights = self.compare_year_over_year(filtered_df, category)
                temporal_insights.extend(yoy_insights)
            
            # Category shifts over time for larger datasets
            if len(filtered_df) >= 100 and not category:
                shift_insights = self.detect_category_shifts(filtered_df)
                temporal_insights.extend(shift_insights)
        
        # Add temporal insights to message
        if temporal_insights:
            message_parts.append("I noticed these patterns over time:")
            for insight in temporal_insights:
                message_parts.append(f"- {insight}")

        if len(filtered_df) >= 5:
            # Group by day for time series analysis
            filtered_df['date'] = filtered_df['timestamp'].dt.date
            daily_counts = filtered_df.groupby('date').size()

            # Check for spikes
            if len(daily_counts) > 3:
                mean = daily_counts.mean()
                std = daily_counts.std()
                threshold = mean + 1.5 * std

                spikes = daily_counts[daily_counts > threshold]
                if not spikes.empty:
                    if len(spikes) == 1:
                        spike_date = spikes.index[0]
                        message_parts.append(f"I noticed a significant spike on {spike_date}.")
                    else:
                        spike_dates = ", ".join([str(date) for date in spikes.index])
                        message_parts.append(f"I noticed significant spikes on these dates: {spike_dates}.")

            # Create time series chart
            charts.append({
                'type': 'time_series',
                'data': daily_counts
            })

        # Create final response
        response = {
            'message': " ".join(message_parts),
            'data': {
                'count': message_count,
                'unique_users': unique_users,
                'charts': charts
            }
        }

        return response
    
    def detect_seasonal_patterns(self, filtered_df):
        """
        Detect seasonal patterns in user feedback
        """
        # Group by day of week
        filtered_df['day_of_week'] = filtered_df['timestamp'].dt.day_name()
        day_counts = filtered_df.groupby('day_of_week').size()
        
        # Group by month
        filtered_df['month'] = filtered_df['timestamp'].dt.month_name()
        month_counts = filtered_df.groupby('month').size()
        
        # Group by hour of day
        filtered_df['hour'] = filtered_df['timestamp'].dt.hour
        hour_counts = filtered_df.groupby('hour').size()
        
        # Compare distribution to check for patterns
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = day_counts.reindex(days_order)
        
        # Check for weekend vs weekday differences
        weekday_avg = day_counts.iloc[:5].mean()
        weekend_avg = day_counts.iloc[5:].mean()
        weekend_ratio = weekend_avg / weekday_avg if weekday_avg > 0 else 0
        
        # Check for monthly patterns (peak months)
        months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                        'July', 'August', 'September', 'October', 'November', 'December']
        month_counts = month_counts.reindex(months_order)
        
        # Return insights
        patterns = []
        if weekend_ratio < 0.7:
            patterns.append(f"Weekdays have {(1-weekend_ratio)*100:.1f}% more messages than weekends")
        elif weekend_ratio > 1.3:
            patterns.append(f"Weekends have {(weekend_ratio-1)*100:.1f}% more messages than weekdays")
        
        # Detect peak months (if significantly higher than average)
        month_mean = month_counts.mean()
        month_std = month_counts.std()
        peak_months = month_counts[month_counts > month_mean + month_std].index.tolist()
        if peak_months:
            patterns.append(f"Peak activity during: {', '.join(peak_months)}")
        
        # Detect hourly patterns
        busy_hours = hour_counts[hour_counts > hour_counts.mean() + hour_counts.std()].index.tolist()
        if busy_hours:
            busy_hours_fmt = [f"{h}:00" for h in busy_hours]
            patterns.append(f"Busiest hours: {', '.join(busy_hours_fmt)}")
        
        return patterns

    def compare_year_over_year(self, filtered_df, category=None):
        """
        Compare current year's data with previous years for the same time period
        """
        if filtered_df.empty:
            return []
        
        # Extract year and month
        filtered_df['year'] = filtered_df['timestamp'].dt.year
        filtered_df['month'] = filtered_df['timestamp'].dt.month
        
        # Group by year and month
        if category:
            category_df = filtered_df[filtered_df['category'] == category]
            year_month_counts = category_df.groupby(['year', 'month']).size().unstack(level=0, fill_value=0)
        else:
            year_month_counts = filtered_df.groupby(['year', 'month']).size().unstack(level=0, fill_value=0)
        
        # Calculate year-over-year growth
        insights = []
        years = sorted(year_month_counts.columns)
        
        if len(years) > 1:
            current_year = years[-1]
            prev_year = years[-2]
            
            # Calculate YoY growth for each month where data exists for both years
            for month in range(1, 13):
                if month in year_month_counts.index and not year_month_counts.empty:
                    if current_year in year_month_counts.columns and prev_year in year_month_counts.columns:
                        current = year_month_counts.loc[month, current_year]
                        previous = year_month_counts.loc[month, prev_year]
                        
                        if previous > 0:
                            growth = ((current - previous) / previous) * 100
                            month_name = datetime(2000, month, 1).strftime('%B')
                            
                            if abs(growth) > 20:  # Only report significant changes
                                direction = "increase" if growth > 0 else "decrease"
                                insights.append(f"{month_name} shows a {abs(growth):.1f}% {direction} compared to last year")
        
        return insights
    
    def detect_trends(self, filtered_df, window=7):
        """
        Detect trends in message volume over time
        """
        if len(filtered_df) < window * 3:
            return []  # Not enough data
        
        # Group by date
        filtered_df['date'] = filtered_df['timestamp'].dt.date
        daily_counts = filtered_df.groupby('date').size()
        
        # Sort by date
        daily_counts = daily_counts.sort_index()
        
        # Calculate rolling average
        rolling_avg = daily_counts.rolling(window=window).mean()
        
        # Calculate slope of trend line
        import numpy as np
        from scipy import stats
        
        # Need sufficient data points
        if len(rolling_avg.dropna()) > 10:
            x = np.arange(len(rolling_avg.dropna()))
            y = rolling_avg.dropna().values
            slope, _, _, _, _ = stats.linregress(x, y)
            
            # Interpret the trend
            insights = []
            if slope > 0.1:  # Positive trend
                percent_increase = (rolling_avg.iloc[-1] / rolling_avg.iloc[window] - 1) * 100
                insights.append(f"Increasing trend detected: ~{percent_increase:.1f}% increase over the time period")
            elif slope < -0.1:  # Negative trend
                percent_decrease = (1 - rolling_avg.iloc[-1] / rolling_avg.iloc[window]) * 100
                insights.append(f"Decreasing trend detected: ~{percent_decrease:.1f}% decrease over the time period")
            
            return insights
        
        return []

    def detect_category_shifts(self, filtered_df):
        """
        Detect shifts in category distribution over time
        """
        # Need at least 90 days of data
        if filtered_df['timestamp'].max() - filtered_df['timestamp'].min() < pd.Timedelta(days=90):
            return []
        
        # Split the data into first half and second half
        filtered_df = filtered_df.sort_values('timestamp')
        midpoint = filtered_df['timestamp'].min() + (filtered_df['timestamp'].max() - filtered_df['timestamp'].min()) / 2
        
        first_half = filtered_df[filtered_df['timestamp'] < midpoint]
        second_half = filtered_df[filtered_df['timestamp'] >= midpoint]
        
        # Calculate category distribution for each half
        first_half_cats = first_half['category'].value_counts(normalize=True)
        second_half_cats = second_half['category'].value_counts(normalize=True)
        
        # Find categories with significant shifts
        insights = []
        for category in set(first_half_cats.index) | set(second_half_cats.index):
            first_pct = first_half_cats.get(category, 0) * 100
            second_pct = second_half_cats.get(category, 0) * 100
            
            # Calculate absolute and relative change
            abs_change = second_pct - first_pct
            rel_change = (second_pct / first_pct - 1) * 100 if first_pct > 0 else float('inf')
            
            # Report significant shifts
            if abs(abs_change) >= 5 and abs(rel_change) >= 20:
                direction = "increased" if abs_change > 0 else "decreased"
                category_name = self.category_display_names.get(category, category.replace('_', ' ').title())
                insights.append(f"{category_name} has {direction} from {first_pct:.1f}% to {second_pct:.1f}% of messages")
        
        return insights

    def _get_context_description(self, context):
        """Get a human-readable description of a context"""
        desc_parts = []

        if context['category']:
            category_name = self.category_display_names.get(
                context['category'],
                context['category'].replace('_', ' ').title()
            )
            desc_parts.append(f"{category_name}")

        if context['source']:
            desc_parts.append(f"from {context['source'].title()}")

        if context['time_range']:
            start_time, end_time = context['time_range']
            time_desc = f"between {start_time.strftime('%Y-%m-%d')} and {end_time.strftime('%Y-%m-%d')}"
            desc_parts.append(time_desc)

        if not desc_parts:
            return "all feedback messages"

        return " ".join(desc_parts)

    def get_context_history(self):
        """
        Get a human-readable summary of the context history

        Returns:
            list: Descriptions of past contexts
        """
        context_descriptions = []

        # Current context first
        current_desc = f"Current context: {self._get_context_description(self.conversation_memory['current_context'])}"
        context_descriptions.append(current_desc)

        # Past contexts
        for i, context in enumerate(self.conversation_memory['filter_history']):
            # Skip empty contexts
            if not any(context.get(key) for key in ['category', 'source', 'time_range']):
                continue

            desc = f"Context {i+1}: {self._get_context_description(context)}"
            context_descriptions.append(desc)

        return context_descriptions

    def _check_context_relation(self, query):
        """
        Check if the query relates to any past contexts beyond the immediate last one

        Returns:
            dict: The most relevant previous context, or None if not found
        """
        # Skip if no history or only one item
        if len(self.conversation_memory['filter_history']) <= 1:
            return None

        query_lower = query.lower()

        # Look for phrases that might reference earlier contexts
        reference_phrases = [
            'earlier', 'before', 'previously', 'you mentioned',
            'go back to', 'return to', 'like earlier', 'last time',
            'first query', 'original question'
        ]

        # Check if query contains reference to earlier context
        if any(phrase in query_lower for phrase in reference_phrases):
            # Check for numeric references like "2 queries ago"
            numeric_match = re.search(r'(\d+)\s*(quer|question|context)s?\s*ago', query_lower)
            if numeric_match:
                index = int(numeric_match.group(1))
                if 0 < index < len(self.conversation_memory['filter_history']):
                    return self.conversation_memory['filter_history'][index]

            # Return the second most recent context (index 1, since index 0 is the current one)
            return self.conversation_memory['filter_history'][1]

        return None

    def get_conversation_history(self):
        """Return the full conversation history"""
        return self.conversation_memory['messages']

