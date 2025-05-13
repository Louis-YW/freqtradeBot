def scrape_by_decreasing_id(self, start_topic_id: str, start_date: str, end_date: str, max_topics: int = 1000):
    """
    Crawl topics by decreasing the topic ID

    Args:
        start_topic_id: The starting topic ID
        start_date: The start date, in the format of YYYY-MM-DD
        end_date: The end date, in the format of YYYY-MM-DD
        max_topics: The maximum number of topics to crawl, with a default value of 1000
    """
    # First, try to log in
    if not self.login():
        logging.error("Failed to login. Cannot proceed with scraping.")
        return
        
    # Convert string dates to datetime objects for date comparison
    try:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        logging.info(f"Date range set: {start_date} to {end_date}")
    except ValueError as e:
        logging.error(f"Invalid date format: {e}. Use YYYY-MM-DD format.")
        return
        
    logging.info(f"Starting to scrape from topic {start_topic_id} using decreasing ID method")
    current_topic_id = start_topic_id
    topics_processed = 0
    topics_saved = 0
    
    # Load the current topic ID from checkpoint.json
    checkpoint_file = "checkpoint.json"
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                if 'current_topic_id' in checkpoint_data:
                    current_topic_id = checkpoint_data.get('current_topic_id')
                    topics_processed = checkpoint_data.get('topics_processed', 0)
                    topics_saved = checkpoint_data.get('topics_saved', 0)
            logging.info(f"Loaded checkpoint: starting from topic {current_topic_id}, processed {topics_processed}, saved {topics_saved}")
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
    
    # Ensure the topic ID is a numeric format
    try:
        numeric_id = int(current_topic_id)
    except ValueError:
        logging.error(f"Topic ID {current_topic_id} is not a numeric format")
        return
    
    while topics_processed < max_topics:
        # Parse the current topic
        topic_data = self.parse_topic(current_topic_id)
        topics_processed += 1
        
        if topic_data:
            # Check if the topic is within the date range
            try:
                # Try to parse the creation time string to a datetime object
                # Example format: "March 24, 2025, 10:30:45 AM"
                created_time_str = topic_data['created_time']
                # Extract the date part
                date_part = created_time_str.split(',')[0] + ',' + created_time_str.split(',')[1]
                topic_date = datetime.strptime(date_part.strip(), "%B %d, %Y")
                
                # Check if the topic date is within the specified range
                if start_date_obj <= topic_date <= end_date_obj:
                    # Check if it contains keywords
                    if self.contains_keywords(topic_data['title'] + topic_data['content']):
                        filename = f"data/json/topic_{current_topic_id}.json"
                        self.save_data(topic_data, filename)
                        topics_saved += 1
                        logging.info(f"Saved topic {current_topic_id} ({topics_saved} total) - Published at: {topic_data['created_time']}")
                else:
                    logging.info(f"Topic {current_topic_id} is outside date range: {topic_date.strftime('%Y-%m-%d')}")
            except Exception as e:
                logging.warning(f"Error parsing date for topic {current_topic_id}: {str(e)}")
                # If the date cannot be parsed, still save the topic
                if self.contains_keywords(topic_data['title'] + topic_data['content']):
                    filename = f"data/json/topic_{current_topic_id}.json"
                    self.save_data(topic_data, filename)
                    topics_saved += 1
                    logging.info(f"Saved topic {current_topic_id} ({topics_saved} total) - Date parsing failed")
        else:
            logging.error(f"Failed to parse topic {current_topic_id}")
        
        # Decrease the topic ID
        try:
            numeric_id = int(current_topic_id)
            numeric_id -= 1  # Decrement ID
            current_topic_id = str(numeric_id)
            logging.info(f"Moving to previous topic: {current_topic_id}")
        except ValueError:
            logging.error(f"Could not decrement topic ID {current_topic_id} as it's not a numeric format")
            logging.info("Stopping due to invalid topic ID format")
            break
        
        # Update and save checkpoint
        try:
            checkpoint_data = {
                "current_topic_id": current_topic_id,
                "topics_processed": topics_processed,
                "topics_saved": topics_saved,
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open("checkpoint.json", 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved checkpoint: current topic {current_topic_id}, processed {topics_processed}, saved {topics_saved}")
        except Exception as e:
            logging.error(f"Error saving checkpoint: {str(e)}")
        
        self.random_sleep(2.0, 5.0)  # Increase waiting time to avoid too frequent requests
    
    logging.info(f"Finished scraping by decreasing ID. Total topics processed: {topics_processed}, saved: {topics_saved}")