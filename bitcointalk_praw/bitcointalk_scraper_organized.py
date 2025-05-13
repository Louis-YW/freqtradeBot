import requests
from bs4 import BeautifulSoup
import json
import time
import random
from datetime import datetime
import logging
from typing import List, Dict, Optional
import re
from fake_useragent import UserAgent
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# Configure logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bitcointalk_scraper.log'),
        logging.StreamHandler()
    ]
)

class BitcointalkScraper:
    def __init__(self, username: str, password: str):
        self.base_url = "https://bitcointalk.org"
        self.ua = UserAgent()
        self.username = username
        self.password = password
        self.keywords = [
            # Basic keywords
            "Bitcoin", "BTC", "Bitcoin Price", "BTC Price", "Bitcoin Value", "BTC Value",
            # Market Trends
            "Bitcoin Trend", "BTC Trend", "Bitcoin Analysis", "BTC Analysis",
            # Price Changes
            "Bitcoin Surge", "BTC Surge", "Bitcoin Crash", "BTC Crash",
            # Technical Indicators
            "Bitcoin RSI", "BTC RSI", "Bitcoin MACD", "BTC MACD",
            # Market sentiment
            "Bitcoin FUD", "BTC FUD", "Bitcoin News", "BTC News",
            # Macroeconomics
            "Bitcoin Inflation", "BTC Inflation", "Bitcoin Interest Rate", "BTC Interest Rate",
            # Historical Events
            "Bitcoin Halving", "BTC Halving", "Bitcoin Mt. Gox", "BTC Mt. Gox"
        ]
        self.keywords = [k.lower() for k in self.keywords]
        
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data/json", exist_ok=True)
        os.makedirs("data/debug_html", exist_ok=True)
        
        # Initialize the Selenium WebDriver
        options = webdriver.ChromeOptions()
        # Remove headless mode to make the browser window visible, facilitating manual captcha resolution
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 10)

    def login(self) -> bool:
        """Use Selenium to log in to the forum"""
        try:
            # Visit the login page
            login_url = f"{self.base_url}/index.php?action=login"
            logging.info(f"Fetching login page: {login_url}")
            self.driver.get(login_url)
            
            # Wait for the login form to load
            username_field = self.wait.until(
                EC.presence_of_element_located((By.NAME, "user"))
            )
            password_field = self.driver.find_element(By.NAME, "passwrd")
            
            # Enter username and password
            username_field.send_keys(self.username)
            password_field.send_keys(self.password)
            
            # Wait for the verification code to load
            try:
                recaptcha = self.wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, "g-recaptcha"))
                )
                logging.info("Found reCAPTCHA, waiting for manual solve...")
                # Wait for the user to manually solve the CAPTCHA
                input("Please solve the CAPTCHA and press Enter to continue...")
                
                # Add extra waiting time to ensure that the verification code has been processed
                time.sleep(3)
                
                # Save the source code of the login page for debugging
                with open("logs/login_page.html", "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
            except TimeoutException:
                logging.info("No CAPTCHA found, proceeding with login...")
            
            # Click the login button
            login_button = self.driver.find_element(By.CSS_SELECTOR, "input[type='submit']")
            login_button.click()
            
            # Increase the waiting time to ensure the login process is completed
            time.sleep(8)
            
            # Save the login response page for debugging
            with open("logs/login_response.html", "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            
            # Check if the login was successful
            if 'action=logout' in self.driver.page_source:
                logging.info("Successfully logged in")
                return True
            else:
                # Check if there is any captcha error message
                if "You must solve the CAPTCHA" in self.driver.page_source:
                    logging.error("Login failed: CAPTCHA was not solved correctly")
                else:
                    logging.error("Login failed: Unknown reason")
                return False
                
        except Exception as e:
            logging.error(f"Error during login: {str(e)}")
            return False

    def fetch_page(self, url: str, max_retries: int = 3) -> Optional[str]:
        """Use Selenium to get page content, and support automatic retry when the session expires"""
        retries = 0
        while retries < max_retries:
            try:
                logging.info(f"Fetching page: {url} (attempt {retries + 1}/{max_retries})")
                self.driver.get(url)
                time.sleep(2)  # Wait for the page to load
                return self.driver.page_source
            except Exception as e:
                error_message = str(e)
                logging.error(f"Error fetching {url}: {error_message}")
                
                # Check if it is a session expiration error
                if "invalid session id" in error_message.lower():
                    logging.warning("Session expired, reinitializing WebDriver and attempting to login again")
                    self.reinitialize_driver()
                    if not self.login():
                        logging.error("Failed to re-login after session expired. Giving up.")
                        return None
                    retries += 1
                    continue
                else:
                    # For other types of errors, increase the number of retries.
                    retries += 1
                    if retries < max_retries:
                        wait_time = 2 ** retries  # Exponential backoff strategy
                        logging.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"Failed to fetch {url} after {max_retries} attempts")
                        return None
        return None
        
    def reinitialize_driver(self):
        """Re - initialize the WebDriver instance"""
        logging.info("Reinitializing WebDriver instance")
        try:
            # Close the existing WebDriver instance
            if hasattr(self, 'driver'):
                try:
                    self.driver.quit()
                except Exception as e:
                    logging.warning(f"Error closing existing WebDriver: {str(e)}")
            
            # Create a new WebDriver instance
            options = webdriver.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            self.driver = webdriver.Chrome(options=options)
            self.wait = WebDriverWait(self.driver, 10)
            logging.info("Successfully reinitialized WebDriver instance")
            return True
        except Exception as e:
            logging.error(f"Failed to reinitialize WebDriver: {str(e)}")
            return False

    def scrape_board(self, board_id: str, start_date: str, end_date: str, sort_by: str = "last_post"):
        """Crawl posts in the specified section
        
        Args:
            board_id: Plate ID
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            sort_by: last_post(Last reply time), starter(Posting time)
        """
        # Try logging in first
        if not self.login():
            logging.error("Failed to login. Cannot proceed with scraping.")
            return
            
        # Convert the string date to a datetime object for date comparison
        try:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            logging.info(f"Date range set: {start_date} to {end_date}")
        except ValueError as e:
            logging.error(f"Invalid date format: {e}. Use YYYY-MM-DD format.")
            return
            
        logging.info(f"Starting to scrape board {board_id} with sort_by={sort_by}")
        page = 0
        topics_processed = 0
        topics_filtered = 0
        
        while True:
            # Build the URL according to the sorting method
            # Optional sorting methods: subject (title), starter (thread starter), replies (number of replies), views (number of views), last_post (time of last reply)
            # Add the parameter desc=1 to make the sorting in descending order (from new to old).
            url = f"{self.base_url}/index.php?board={board_id}.{page*40};sort={sort_by};desc=1"
            logging.info(f"Fetching page {page + 1} from board {board_id} with sort={sort_by} (newest first)")
            
            # Use the fetch_page method with session reinitialization to get page content
            html = self.fetch_page(url, max_retries=3)
            if not html:
                logging.error(f"Failed to fetch page {page + 1} from board {board_id}")
                # Do not immediately interrupt, but try to continue to the next page
                page += 1
                self.random_sleep(5.0, 10.0)  # Increase waiting time
                continue

            soup = BeautifulSoup(html, 'html.parser')
            
            # Check if you need to re-login
            if soup.find('form', {'id': 'frmLogin'}):
                logging.error("Session expired, attempting to re-login")
                if not self.login():
                    logging.error("Failed to re-login. Stopping scraping.")
                    break
                continue
            
            # Save the page source code for debugging
            with open("logs/debug_response.html", "w", encoding="utf-8") as f:
                f.write(html)
            logging.info(f"Saved response to logs/debug_response.html")
            
            # Find the list of topics
            # In the debug_response.html file, observe that the topics are in td elements with the class windowbg, not tr elements
            topics = soup.find_all('tr')
            topic_rows = []
            for tr in topics:
                # Find the td element containing the topic link
                subject_td = tr.find('td', {'class': 'windowbg'})
                if subject_td and subject_td.find('a', href=lambda href: href and 'topic=' in href):
                    topic_rows.append(tr)
                    
            logging.info(f"Found {len(topic_rows)} potential topics on page {page + 1}")
            
            if not topic_rows:
                logging.info(f"No more topics found on page {page + 1}")
                break
                
            topics = topic_rows

            for topic in topics:
                try:
                    # Find the td element containing the topic link
                    subject_td = topic.find('td', {'class': 'windowbg'})
                    if not subject_td:
                        continue
                        
                    # Find the topic link and extract topic_id
                    topic_link = subject_td.find('a', href=lambda href: href and 'topic=' in href)
                    if not topic_link:
                        continue
                        
                    # Extract topic_id from the link, format: https://bitcointalk.org/index.php?topic=5532596.0
                    href = topic_link['href']
                    topic_id = href.split('topic=')[1].split('.')[0]
                    topic_data = self.parse_topic(topic_id)
                    
                    if topic_data and self.contains_keywords(topic_data['title'] + topic_data['content']):
                        filename = f"data/json/topic_{topic_id}.json"
                        self.save_data(topic_data, filename)
                        topics_processed += 1
                        logging.info(f"Processed topic {topic_id} ({topics_processed} total) - Published at: {topic_data['created_time']}")
                    
                    self.random_sleep()
                except Exception as e:
                    logging.error(f"Error processing topic: {str(e)}")
                    continue

            # Update and save checkpoint
            page += 1
            try:
                checkpoint_data = {
                    "current_page": page,
                    "topics_processed": topics_processed,
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                with open("checkpoint.json", 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                logging.info(f"Saved checkpoint: page {page}, processed {topics_processed} topics")
            except Exception as e:
                logging.error(f"Error saving checkpoint: {str(e)}")
                
            self.random_sleep()
        
        logging.info(f"Finished scraping board {board_id}. Total topics processed: {topics_processed}")

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'driver'):
            self.driver.quit()

    def random_sleep(self, min_seconds: float = 1.0, max_seconds: float = 5.0):
        """Randomly wait for a period of time"""
        time.sleep(random.uniform(min_seconds, max_seconds))

    def contains_keywords(self, text: str) -> bool:
        """Check if the text contains keywords"""
        text = text.lower()
        return any(keyword in text for keyword in self.keywords)

    def parse_topic(self, topic_id: str, max_retries: int = 3) -> Optional[Dict]:
        """Parse a single post, support automatic retry when the session expires"""
        url = f"{self.base_url}/index.php?topic={topic_id}"
        logging.info(f"Parsing topic: {topic_id}")
        
        # Use the fetch_page method with session reinitialization to get page content
        html = self.fetch_page(url, max_retries)
        if not html:
            logging.error(f"Failed to fetch topic {topic_id}")
            return None

        soup = BeautifulSoup(html, 'html.parser')
        
        # Save the topic page source code for debugging
        debug_html_path = f"data/debug_html/topic_{topic_id}_debug.html"
        with open(debug_html_path, "w", encoding="utf-8") as f:
            f.write(html)
        logging.info(f"Saved topic page to {debug_html_path}")
        
        try:
            # In the bitcointalk forum, the topic title is usually in the span element, not the h1 element
            title_span = soup.find('span', id=lambda x: x and x.startswith('msg_'))
            if title_span and title_span.find('a'):
                title = title_span.find('a').text.strip()
            else:
                # Try other possible title locations
                title_element = soup.find('td', {'class': 'windowbg'}) or soup.find('td', {'class': 'windowbg2'})
                if title_element and title_element.find('b'):
                    title = title_element.find('b').text.strip()
                else:
                    logging.error(f"Could not find title for topic {topic_id}")
                    return None
            
            # Find the author information
            poster_info = soup.find('td', {'class': 'poster_info'})
            if poster_info and poster_info.find('b'):
                author = poster_info.find('b').text.strip()
            else:
                logging.error(f"Could not find author for topic {topic_id}")
                return None
            
            # Find the creation time
            header_post = soup.find('td', {'class': 'td_headerandpost'})
            if header_post and header_post.find('div', {'class': 'smalltext'}):
                created_time = header_post.find('div', {'class': 'smalltext'}).text.strip()
            else:
                logging.error(f"Could not find creation time for topic {topic_id}")
                return None
            
            # Find the main post content
            post_div = soup.find('div', {'class': 'post'})
            if post_div:
                content = post_div.text.strip()
            else:
                logging.error(f"Could not find content for topic {topic_id}")
                return None
            
            # Try to get the like and dislike data
            # In bitcointalk, this information may be in specific elements
            score = 0
            upvotes = 0
            downvotes = 0
            
            # Try to find the like and dislike information
            vote_div = soup.find('div', {'class': 'vote'})
            if vote_div:
                # Parse the like and dislike information
                upvote_span = vote_div.find('span', {'class': 'upvote'})
                downvote_span = vote_div.find('span', {'class': 'downvote'})
                
                if upvote_span and 'data-count' in upvote_span.attrs:
                    upvotes = int(upvote_span['data-count'])
                if downvote_span and 'data-count' in downvote_span.attrs:
                    downvotes = int(downvote_span['data-count'])
                
                score = upvotes - downvotes
            
            comments = []
            for comment in soup.find_all('div', {'class': 'post'}):
                if comment == soup.find('div', {'class': 'post'}):  # Skip the main post
                    continue
                
                try:
                    poster_info = comment.find_previous('td', {'class': 'poster_info'})
                    header_post = comment.find_previous('td', {'class': 'td_headerandpost'})
                    
                    if poster_info and poster_info.find('b') and header_post and header_post.find('div', {'class': 'smalltext'}):
                        # Try to get the like and dislike information of the comment
                        comment_upvotes = 0
                        comment_downvotes = 0
                        comment_score = 0
                        
                        comment_vote_div = comment.find_previous('div', {'class': 'vote'})
                        if comment_vote_div:
                            comment_upvote_span = comment_vote_div.find('span', {'class': 'upvote'})
                            comment_downvote_span = comment_vote_div.find('span', {'class': 'downvote'})
                            
                            if comment_upvote_span and 'data-count' in comment_upvote_span.attrs:
                                comment_upvotes = int(comment_upvote_span['data-count'])
                            if comment_downvote_span and 'data-count' in comment_downvote_span.attrs:
                                comment_downvotes = int(comment_downvote_span['data-count'])
                            
                            comment_score = comment_upvotes - comment_downvotes
                        
                        comment_data = {
                            'author': poster_info.find('b').text.strip(),
                            'created_time': header_post.find('div', {'class': 'smalltext'}).text.strip(),
                            'body': comment.text.strip(),
                            'score': comment_score,
                            'upvotes': comment_upvotes,
                            'downvotes': comment_downvotes
                        }
                        comments.append(comment_data)
                    else:
                        logging.warning(f"Skipping comment due to missing elements")
                except Exception as e:
                    logging.warning(f"Error parsing comment: {str(e)}")
                    continue

            topic_data = {
                'id': f"topic_{topic_id}",
                'title': title,
                'author': author,
                'created_time': created_time,
                'content': content,
                'score': score,
                'upvotes': upvotes,
                'downvotes': downvotes,
                'url': url,
                'comments': comments
            }
            
            logging.info(f"Successfully parsed topic {topic_id}")
            # Save the parsed data for debugging
            parsed_json_path = f"data/json/topic_{topic_id}_parsed.json"
            with open(parsed_json_path, "w", encoding="utf-8") as f:
                json.dump(topic_data, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved parsed data to {parsed_json_path}")
            
            return topic_data
            
        except Exception as e:
            logging.error(f"Error parsing topic {topic_id}: {str(e)}")
            return None

    def save_data(self, data: Dict, filename: str):
        """Save data to a JSON file"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logging.info(f"Data saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving data to {filename}: {str(e)}")

    # Removed the scrape_by_next_topic method, because we only use the decreasing ID method to scrape
    
    def scrape_by_decreasing_id(self, start_topic_id: str, start_date: str, end_date: str, max_topics: int = 1000):
        """
        Scrape topics by decreasing ID
        
        Args:
            start_topic_id: The starting topic ID
            start_date: The starting date, format: YYYY-MM-DD
            end_date: The ending date, format: YYYY-MM-DD
            max_topics: The maximum number of topics to scrape, default is 1000
        """
        # Try to login first
        if not self.login():
            logging.error("Failed to login. Cannot proceed with scraping.")
            return
            
        # Convert the string date to a datetime object for date comparison
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
                # Check if the topic is in the date range
                try:
                    # Try to parse the creation time string to a datetime object
                    # Example format: "March 24, 2025, 10:30:45 AM"
                    created_time_str = topic_data['created_time']
                    # Extract the date part
                    date_part = created_time_str.split(',')[0] + ',' + created_time_str.split(',')[1]
                    topic_date = datetime.strptime(date_part.strip(), "%B %d, %Y")
                    
                    # Check if the topic date is in the specified range
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
                numeric_id -= 1  # Decrease ID
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
    
    def get_next_topic_id(self, current_topic_id: str) -> Optional[str]:
        """Get the ID of the next topic
        
        Args:
            current_topic_id: The current topic ID
            
        Returns:
            The ID of the next topic, or None if not found
        """
        url = f"{self.base_url}/index.php?topic={current_topic_id}"
        html = self.fetch_page(url)
        if not html:
            return None
            
        soup = BeautifulSoup(html, 'html.parser')
        
        # Save HTML for debugging
        debug_file = f"data/debug_html/topic_{current_topic_id}_debug.html"
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(html)
        logging.debug(f"Saved HTML to {debug_file} for debugging")
        
        # Find the navigation link area - first try to find the div.nav element
        nav_div = soup.find('div', {'class': 'nav'})
        if nav_div:
            logging.debug(f"Found navigation div: {nav_div}")
            # Find the 'next topic' link in the navigation div
            next_topic_link = nav_div.find('a', text=lambda text: text and 'next topic' in text.lower())
            if next_topic_link:
                logging.debug(f"Found next topic link in nav div: {next_topic_link.get_text()} - {next_topic_link['href']}")
            else:
                # If the exact text is not found, try to find all links
                nav_links = nav_div.find_all('a')
                logging.debug(f"Found {len(nav_links)} links in nav div")
                for link in nav_links:
                    logging.debug(f"Nav div link: {link.get_text()} - {link.get('href')}")
                    if 'prev_next=next' in link.get('href', ''):
                        next_topic_link = link
                        logging.debug(f"Found next topic link by href in nav div: {link.get_text()} - {link['href']}")
                        break
        else:
            logging.debug("Nav div not found, trying alternative methods")
            # If the navigation div is not found, fall back to the original method
            # Find all navigation links
            nav_links = soup.find_all('a', href=lambda href: href and ('prev_next=prev' in href or 'prev_next=next' in href))
            logging.debug(f"Found {len(nav_links)} navigation links")
            for link in nav_links:
                logging.debug(f"Navigation link: {link.get_text()} - {link['href']}")
            
            # Find the 'next topic' link - try multiple methods
            # Method 1: Find through href attribute
            next_topic_link = soup.find('a', href=lambda href: href and 'prev_next=next' in href)
            
            # Method 2: If method 1 fails, try to find through link text
            if not next_topic_link:
                for link in soup.find_all('a'):
                    if link.get_text().strip().lower() in ['next topic', 'next', 'next topic']:
                        next_topic_link = link
                        logging.debug(f"Found next topic link by text: {link.get_text()} - {link.get('href')}")
                        break
        
        if next_topic_link:
            href = next_topic_link['href']
            logging.info(f"Found next topic link: {href}")
            
            # Extract the topic_id from the link, format: https://bitcointalk.org/index.php?topic=5532596.0;prev_next=next#new
            try:
                # Ensure the link contains the topic parameter
                if 'topic=' not in href:
                    logging.warning(f"Next topic link does not contain topic parameter: {href}")
                    return None
                    
                # Extract the topic_id
                new_topic_id = href.split('topic=')[1].split('.')[0]
                
                # Ensure the link points to a different topic
                if new_topic_id == current_topic_id:
                    logging.warning(f"Next topic link points to the same topic: {current_topic_id}")
                    return None
                    
                logging.info(f"Extracted next topic ID: {new_topic_id}")
                return new_topic_id
            except Exception as e:
                logging.error(f"Error extracting topic ID from link {href}: {str(e)}")
                return None
        else:
            logging.info(f"No 'next topic' link found for topic {current_topic_id}")
            
            # Try to find other possible navigation links
            other_links = []
            for link in soup.find_all('a'):
                text = link.get_text().strip().lower()
                if text and ('next' in text or 'next topic' in text):
                    other_links.append((text, link.get('href')))
            
            if other_links:
                logging.info(f"Found {len(other_links)} potential navigation links:")
                for text, href in other_links:
                    logging.info(f"  - {text}: {href}")
            
            return None

def main():
    try:
        # Get username and password from environment variables
        username = os.getenv('BITCOINTALK_USERNAME')
        password = os.getenv('BITCOINTALK_PASSWORD')
        
        if not username or not password:
            logging.error("Please set BITCOINTALK_USERNAME and BITCOINTALK_PASSWORD environment variables")
            return
            
        scraper = BitcointalkScraper(username, password)
        
        # Set date range
        start_date = "2018-01-01"
        end_date = "2025-03-24"
        
        # Use the decreasing ID method to scrape, starting from the specified topic ID
        start_topic_id = "5411596"  # Starting topic ID
        max_topics = 100000  # Maximum number of topics to scrape
        
        logging.info(f"Starting to scrape from topic {start_topic_id} from {start_date} to {end_date}")
        scraper.scrape_by_decreasing_id(start_topic_id, start_date, end_date, max_topics)
        logging.info("Scraping completed")
    except Exception as e:
        logging.error(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main()