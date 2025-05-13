import os
import json
import re
from datetime import datetime
import pandas as pd
from collections import defaultdict

# Define data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'json')

# List to store date information
date_info = []

# Regular expressions to match different date formats
date_patterns = [
    # 'December 28, 2015, 04:12:47 PM'
    r'([A-Z][a-z]+)\s+(\d{1,2}),\s+(\d{4})',
    # Other possible date formats
    r'(\d{2})-(\d{2})-(\d{4})',
    r'(\d{4})-(\d{2})-(\d{2})'
]

# Mapping from month names to numbers
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

def parse_date(date_str):
    """Parse date strings of different formats"""
    for pattern in date_patterns:
        match = re.search(pattern, date_str)
        if match:
            if match.group(1) in month_map:
                # Format like 'December 28, 2015, 04:12:47 PM'
                month = month_map[match.group(1)]
                day = int(match.group(2))
                year = int(match.group(3))
                return datetime(year, month, day)
            elif len(match.group(1)) == 4:
                # Format like '2015-12-28'
                year = int(match.group(1))
                month = int(match.group(2))
                day = int(match.group(3))
                return datetime(year, month, day)
            else:
                # Format like '12-28-2015'
                month = int(match.group(1))
                day = int(match.group(2))
                year = int(match.group(3))
                return datetime(year, month, day)
    return None

def analyze_time_range():
    """Analyze the time range of all JSON files"""
    # Get all JSON files
    json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json') and not f.endswith('_parsed.json')]
    
    print(f"Found {len(json_files)} JSON files")
    
    # Traverse all files
    for filename in json_files:
        file_path = os.path.join(DATA_DIR, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extract topic ID and creation time
                topic_id = data.get('id', '')
                created_time = data.get('created_time', '')
                title = data.get('title', '')
                url = data.get('url', '')
                
                # Parse the date
                date_obj = parse_date(created_time)
                
                if date_obj:
                    date_info.append({
                        'topic_id': topic_id,
                        'date': date_obj,
                        'year': date_obj.year,
                        'month': date_obj.month,
                        'day': date_obj.day,
                        'title': title,
                        'url': url
                    })
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    
    # Convert to DataFrame for analysis
    if date_info:
        df = pd.DataFrame(date_info)
        
        # Sort by date
        df = df.sort_values('date')
        
        # Get the earliest and latest dates
        earliest_date = df['date'].min()
        latest_date = df['date'].max()
        
        print(f"\nData covers the period: {earliest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}")
        
        # Count posts by year and month
        year_month_counts = df.groupby(['year', 'month']).size().reset_index(name='count')
        print("\nNumber of posts by year and month:")
        for _, row in year_month_counts.iterrows():
            print(f"{row['year']}-{row['month']:02d}: {row['count']} posts")
        
        # Show the earliest 5 posts
        print("\nEarliest 5 posts:")
        for _, row in df.head(5).iterrows():
            print(f"{row['date'].strftime('%Y-%m-%d')}: {row['title']} (ID: {row['topic_id']})")
        
        # Show the latest 5 posts
        print("\nLatest 5 posts:")
        for _, row in df.tail(5).iterrows():
            print(f"{row['date'].strftime('%Y-%m-%d')}: {row['title']} (ID: {row['topic_id']})")
    else:
        print("No valid date information found")

if __name__ == "__main__":
    analyze_time_range()
