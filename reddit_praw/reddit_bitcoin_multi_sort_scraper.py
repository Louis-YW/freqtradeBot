#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reddit r/Bitcoin Multi-Sort Scraper

This script can scrape posts from the Reddit r/Bitcoin subreddit using different sorting methods (new, hot, top, controversial, rising), 
supporting checkpointing for batch data retrieval.
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

import praw
import requests
from dotenv import load_dotenv
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper_multi_sort.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RedditBitcoinMultiSortScraper:
    """Reddit r/Bitcoin Multi-Sort Scraper"""

    # Supported sorting methods
    SORT_METHODS = ['new', 'hot', 'top', 'controversial', 'rising']
    
    # Time filters supported for Top and Controversial
    TIME_FILTERS = ['all', 'day', 'week', 'month', 'year']

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        output_dir: str = "data",
        batch_size: int = 1000,
        request_delay: float = 1.0,
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Initialize Reddit Multi-Sort Scraper

        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string
            output_dir: Output directory
            batch_size: Number of posts per batch
            request_delay: Request delay time (seconds)
            checkpoint_dir: Checkpoint directory
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.subreddit = self.reddit.subreddit("Bitcoin")
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.request_delay = request_delay
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create output directory and checkpoint directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def extract_post_data(self, post) -> Dict[str, Any]:
        """
        Extract required data from post object

        Args:
            post: PRAW post object

        Returns:
            Dictionary containing extracted fields
        """
        return {
            "id": post.id,
            "title": post.title,
            "author": str(post.author) if post.author else "[deleted]",
            "created_utc": post.created_utc,
            "created_time": datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            "score": post.score,
            "upvote_ratio": post.upvote_ratio,
            "num_comments": post.num_comments,
            "permalink": post.permalink,
            "url": post.url,
            "is_self": post.is_self,
            "selftext": post.selftext,
            "stickied": post.stickied,
            "over_18": post.over_18,
            "spoiler": post.spoiler,
            "link_flair_text": post.link_flair_text
        }
    
    def save_batch(self, posts: List[Dict[str, Any]], sort_method: str, 
                  time_filter: Optional[str] = None, batch_num: int = 1) -> str:
        """
        Save a batch of post data to a JSON file

        Args:
            posts: List of post data
            sort_method: Sorting method
            time_filter: Time filter (only for top and controversial)
            batch_num: Batch number

        Returns:
            Saved file path
        """
        if not posts:
            return ""
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Construct filename
        if time_filter and sort_method in ['top', 'controversial']:
            filename = f"bitcoin_{sort_method}_{time_filter}_batch_{batch_num}_{timestamp}.json"
        else:
            filename = f"bitcoin_{sort_method}_batch_{batch_num}_{timestamp}.json"
            
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(posts, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(posts)} posts data to {filepath}")
        return str(filepath)
    
    def get_checkpoint_file(self, sort_method: str, time_filter: Optional[str] = None) -> Path:
        """
        Get checkpoint file path

        Args:
            sort_method: Sorting method
            time_filter: Time filter (only for top and controversial)

        Returns:
            Checkpoint file path
        """
        if time_filter and sort_method in ['top', 'controversial']:
            return self.checkpoint_dir / f"{sort_method}_{time_filter}_checkpoint.json"
        else:
            return self.checkpoint_dir / f"{sort_method}_checkpoint.json"
    
    def load_checkpoint(self, sort_method: str, time_filter: Optional[str] = None) -> Tuple[Optional[str], int, int]:
        """
        Load last scraped post ID from checkpoint file

        Args:
            sort_method: Sorting method
            time_filter: Time filter (only for top and controversial)

        Returns:
            (Last scraped post ID, Last batch number, Total scraped posts)
        """
        checkpoint_file = self.get_checkpoint_file(sort_method, time_filter)
        
        if not checkpoint_file.exists():
            logger.info(f"Checkpoint file not found: {checkpoint_file}, starting from scratch")
            return None, 1, 0
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                
            if 'last_post_id' in checkpoint_data:
                last_post_id = checkpoint_data['last_post_id']
                last_batch = checkpoint_data.get('last_batch', 0)
                last_total = checkpoint_data.get('total_posts', 0)
                last_time = checkpoint_data.get('timestamp', 'unknown')
                
                sort_info = f"{sort_method}"
                if time_filter:
                    sort_info += f" ({time_filter})"
                    
                logger.info(f"Continuing from checkpoint {sort_info}, Last batch: {last_batch},"
                           f" Scraped: {last_total} posts, "
                           f"Last post ID: {last_post_id}, "
                           f"Time: {last_time}")
                
                return f"t3_{last_post_id}", last_batch + 1, last_total
            
            logger.warning("Checkpoint file format incorrect, starting from scratch")
            return None, 1, 0
        
        except Exception as e:
            logger.error(f"Error reading checkpoint file: {str(e)}, starting from scratch")
            return None, 1, 0
    
    def save_checkpoint(self, sort_method: str, last_post_id: str, 
                       batch_num: int, total_posts: int, 
                       time_filter: Optional[str] = None) -> None:
        """
        Save checkpoint information

        Args:
            sort_method: Sorting method
            last_post_id: Last post ID
            batch_num: Current batch number
            total_posts: Total number of posts
            time_filter: Time filter (only for top and controversial)
        """
        checkpoint_file = self.get_checkpoint_file(sort_method, time_filter)
        
        checkpoint_data = {
            'last_post_id': last_post_id,
            'last_batch': batch_num,
            'total_posts': total_posts,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        sort_info = f"{sort_method}"
        if time_filter:
            sort_info += f" ({time_filter})"
            
        logger.info(f"Saved checkpoint {sort_info}, Last post ID: {last_post_id},"
                   f" Current batch: {batch_num}, Total posts: {total_posts}")

    def fetch_posts(
        self, 
        sort_method: str,
        time_filter: Optional[str] = None,
        limit_per_request: int = 100,
        max_requests: Optional[int] = None,
        start_after: Optional[str] = None,
        max_posts: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], int, str]:
        """
        Scrape posts by sorting method

        Args:
            sort_method: Sorting method ('new', 'hot', 'top', 'controversial', 'rising')
            time_filter: Time filter (only for top and controversial)
            limit_per_request: Number of posts per request
            max_requests: Maximum number of requests, None means unlimited
            start_after: Starting position for scraping, None means from the beginning
            max_posts: Maximum number of posts to scrape, None means unlimited

        Returns:
            (List of scraped post data, Number of requests made, Last post ID)
        """
        if sort_method not in self.SORT_METHODS:
            raise ValueError(f"Unsupported sorting method: {sort_method}, "
                           f"Supported sorting methods are: {', '.join(self.SORT_METHODS)}")
            
        if sort_method in ['top', 'controversial'] and time_filter and time_filter not in self.TIME_FILTERS:
            raise ValueError(f"Unsupported time filter: {time_filter}, "
                           f"Supported time filters are: {', '.join(self.TIME_FILTERS)}")
        
        all_posts = []
        after = start_after
        request_count = 0
        consecutive_empty_results = 0  # Count of consecutive empty results
        max_consecutive_empty = 3      # Maximum allowed consecutive empty results
        last_post_id = None
        
        # Create sorting method description
        sort_info = f"{sort_method}"
        if time_filter and sort_method in ['top', 'controversial']:
            sort_info += f" ({time_filter})"
            
        start_info = f"Starting from scratch" if after is None else f"Continuing from position {after}"
        logger.info(f"Starting to scrape posts from r/Bitcoin (Sorting: {sort_info}, {start_info})...")
        
        while after is not None or request_count == 0:  # Ensure at least one execution
            try:
                # Check if maximum requests reached
                if max_requests is not None and request_count >= max_requests:
                    logger.info(f"Maximum requests reached ({max_requests}), stopping scrape")
                    break
                    
                # Check if maximum posts reached
                if max_posts is not None and len(all_posts) >= max_posts:
                    logger.info(f"Maximum posts reached ({max_posts}), stopping scrape")
                    break
                
                # Build request parameters
                params = {
                    "limit": limit_per_request
                }
                if after:
                    params["after"] = after
                    
                logger.info(f"Starting {request_count + 1}th request {sort_info}, after={after}")
                
                # Get post list based on sorting method
                if sort_method == "new":
                    listing = self.subreddit.new(limit=limit_per_request, params={"after": after} if after else None)
                elif sort_method == "hot":
                    listing = self.subreddit.hot(limit=limit_per_request, params={"after": after} if after else None)
                elif sort_method == "rising":
                    listing = self.subreddit.rising(limit=limit_per_request, params={"after": after} if after else None)
                elif sort_method == "top":
                    # Get basic top list, without passing after parameter
                    base_listing = self.subreddit.top(time_filter=time_filter or 'all', limit=limit_per_request)
                    
                    if after:
                        # If there's an after value, manually skip some results
                        listing = []
                        seen_after = False
                        for post in base_listing:
                            full_name = f"t3_{post.id}"
                            if full_name == after:
                                seen_after = True
                                continue
                            if seen_after:
                                listing.append(post)
                    else:
                        listing = base_listing
                elif sort_method == "controversial":
                    # Get basic controversial list, without passing after parameter
                    base_listing = self.subreddit.controversial(time_filter=time_filter or 'all', limit=limit_per_request)
                    
                    if after:
                        # If there's an after value, manually skip some results
                        listing = []
                        seen_after = False
                        for post in base_listing:
                            full_name = f"t3_{post.id}"
                            if full_name == after:
                                seen_after = True
                                continue
                            if seen_after:
                                listing.append(post)
                    else:
                        listing = base_listing
                
                # Get current page posts
                current_page_posts = list(listing)
                request_count += 1
                
                # If no posts returned, it might be at the end
                if not current_page_posts:
                    consecutive_empty_results += 1
                    logger.info(f"No posts found in {request_count}th request, consecutive empty results: "
                               f"{consecutive_empty_results}/{max_consecutive_empty}")
                    
                    if consecutive_empty_results >= max_consecutive_empty:
                        logger.info(f"Consecutive {max_consecutive_empty} requests failed, scraping completed")
                        break
                    
                    # If after is None and no results, it's actually at the end
                    if after is None:
                        logger.info("Reached last page (after=None), scraping completed")
                        break
                    
                    # Try adding delay and continue
                    time.sleep(self.request_delay * 2)
                    continue
                
                # Reset consecutive empty results count
                consecutive_empty_results = 0
                
                # Process current page posts
                page_data = []
                
                for post in current_page_posts:
                    try:
                        post_data = self.extract_post_data(post)
                        page_data.append(post_data)
                        # Update last post ID
                        last_post_id = post.id
                    except Exception as e:
                        logger.error(f"Error processing post {post.id}: {str(e)}")
                        continue
                
                # Add to total list
                all_posts.extend(page_data)
                
                logger.info(f"Completed {request_count}th request {sort_info}, got {len(page_data)} posts, "
                          f"Total {len(all_posts)} posts")
                
                # Set next page marker
                if len(current_page_posts) > 0:
                    after = f"t3_{current_page_posts[-1].id}"
                else:
                    after = None
                
                # Add delay to avoid rate limit
                time.sleep(self.request_delay)
                
            except Exception as e:
                logger.error(f"Error in {request_count}th request {sort_info}: {str(e)}")
                # Add delay time and continue
                time.sleep(self.request_delay * 2)
                # Max retry 3 times for the same after value
                consecutive_empty_results += 1
                if consecutive_empty_results >= max_consecutive_empty:
                    logger.warning(f"Consecutive {max_consecutive_empty} requests failed, continuing to next request")
                    # Try skipping current problematic page, get next page
                    if after and len(all_posts) > 0:
                        try:
                            # Try getting new after value based on existing data
                            last_known_id = all_posts[-1]["id"]
                            after = f"t3_{last_known_id}"
                            logger.info(f"Trying new after value: {after}")
                            consecutive_empty_results = 0
                        except Exception:
                            logger.error("Cannot generate new after value, stopping scrape")
                            break
                    else:
                        logger.error("Cannot continue scraping, stopping")
                        break
                continue
        
        logger.info(f"{sort_info} scraping completed. Total {request_count} requests made, "
                    f"scraped {len(all_posts)} posts")
        return all_posts, request_count, last_post_id

    def scrape_by_sort(
        self,
        sort_method: str,
        time_filter: Optional[str] = None,
        limit_per_request: int = 100,
        max_requests: Optional[int] = None,
        max_posts: Optional[int] = None,
        reset: bool = False
    ) -> int:
        """
        Scrape and save posts by sorting method

        Args:
            sort_method: Sorting method ('new', 'hot', 'top', 'controversial', 'rising')
            time_filter: Time filter (only for top and controversial)
            limit_per_request: Number of posts per request
            max_requests: Maximum number of requests, None means unlimited
            max_posts: Maximum number of posts to scrape, None means unlimited
            reset: Whether to reset checkpoint and start scraping from the beginning

        Returns:
            Total number of scraped posts
        """
        # If need to reset checkpoint
        if reset:
            checkpoint_file = self.get_checkpoint_file(sort_method, time_filter)
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                sort_info = f"{sort_method}"
                if time_filter:
                    sort_info += f" ({time_filter})"
                logger.info(f"Reset checkpoint file {sort_info}: {checkpoint_file}")
        
        # Load checkpoint
        start_after, start_batch, previous_total = self.load_checkpoint(sort_method, time_filter)
        
        # Scrape posts by sorting method
        all_posts, request_count, last_post_id = self.fetch_posts(
            sort_method=sort_method,
            time_filter=time_filter,
            limit_per_request=limit_per_request,
            max_requests=max_requests,
            start_after=start_after,
            max_posts=max_posts
        )
        
        # Save posts by batch
        total_saved = 0
        batch_count = start_batch
        
        for i in range(0, len(all_posts), self.batch_size):
            batch = all_posts[i:i+self.batch_size]
            self.save_batch(batch, sort_method, time_filter, batch_count)
            total_saved += len(batch)
            batch_count += 1
        
        # Save checkpoint, record last post ID
        if last_post_id:
            total_saved += previous_total  # Add previous scraped total
            self.save_checkpoint(sort_method, last_post_id, batch_count-1, total_saved, time_filter)
        
        sort_info = f"{sort_method}"
        if time_filter:
            sort_info += f" ({time_filter})"
            
        logger.info(f"{sort_info} scraping completed. Total {request_count} requests made, "
                    f"scraped {len(all_posts)} posts, "
                    f"saved as {batch_count - start_batch} batches.")
        
        return len(all_posts)
        
    def scrape_all_sorts(
        self,
        include_time_filters: bool = True,
        limit_per_request: int = 100,
        max_requests: Optional[int] = None,
        max_posts: Optional[int] = None,
        reset: bool = False
    ) -> Dict[str, int]:
        """
        Scrape posts from all sorting methods

        Args:
            include_time_filters: Whether to include time filters for top and controversial
            limit_per_request: Number of posts per request
            max_requests: Maximum number of requests, None means unlimited
            max_posts: Maximum number of posts to scrape, None means unlimited
            reset: Whether to reset checkpoint and start scraping from the beginning

        Returns:
            Number of posts scraped for each sorting method
        """
        results = {}
        
        # Scrape basic sorting methods
        for sort_method in self.SORT_METHODS:
            # For top and controversial, if time filters are needed
            if sort_method in ['top', 'controversial'] and include_time_filters:
                for time_filter in self.TIME_FILTERS:
                    sort_key = f"{sort_method}_{time_filter}"
                    logger.info(f"Starting to scrape posts from {sort_method} ({time_filter}) sorting...")
                    
                    count = self.scrape_by_sort(
                        sort_method=sort_method,
                        time_filter=time_filter,
                        limit_per_request=limit_per_request,
                        max_requests=max_requests,
                        max_posts=max_posts,
                        reset=reset
                    )
                    
                    results[sort_key] = count
                    logger.info(f"Completed scraping posts from {sort_method} ({time_filter}) sorting: {count} posts")
            else:
                logger.info(f"Starting to scrape posts from {sort_method} sorting...")
                
                count = self.scrape_by_sort(
                    sort_method=sort_method,
                    limit_per_request=limit_per_request,
                    max_requests=max_requests,
                    max_posts=max_posts,
                    reset=reset
                )
                
                results[sort_method] = count
                logger.info(f"Completed scraping posts from {sort_method} sorting: {count} posts")
        
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Reddit r/Bitcoin Multi-Sort Scraper")
    parser.add_argument("--config", help="Configuration file path (.env)", default=".env")
    parser.add_argument("--output", help="Output directory", default="data")
    parser.add_argument("--batch-size", help="Number of posts per batch", type=int, default=1000)
    parser.add_argument("--delay", help="Request delay time (seconds)", type=float, default=1.0)
    parser.add_argument("--limit-per-request", help="Number of posts per request", type=int, default=100)
    parser.add_argument("--max-requests", help="Maximum number of requests", type=int)
    parser.add_argument("--max-posts", help="Maximum number of posts to scrape", type=int)
    parser.add_argument("--sort", help=f"Post sorting method {RedditBitcoinMultiSortScraper.SORT_METHODS}", 
                      choices=RedditBitcoinMultiSortScraper.SORT_METHODS)
    parser.add_argument("--time-filter", help=f"Time filter (only for top and controversial) {RedditBitcoinMultiSortScraper.TIME_FILTERS}", 
                      choices=RedditBitcoinMultiSortScraper.TIME_FILTERS)
    parser.add_argument("--all", action="store_true", help="Scrape posts from all sorting methods")
    parser.add_argument("--include-time-filters", action="store_true", help="Include time filters for top and controversial")
    parser.add_argument("--checkpoint-dir", help="Checkpoint directory", default="checkpoints")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint and start scraping from the beginning")
    args = parser.parse_args()
    
    # Load environment variables
    if os.path.exists(args.config):
        load_dotenv(args.config)
    
    # Get Reddit API credentials
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "r/Bitcoin Multi Sort Scraper by /u/YOURNAME")
    
    if not client_id or not client_secret:
        logger.error("Reddit API credentials not provided. Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env file.")
        return
    
    # Initialize scraper
    scraper = RedditBitcoinMultiSortScraper(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        output_dir=args.output,
        batch_size=args.batch_size,
        request_delay=args.delay,
        checkpoint_dir=args.checkpoint_dir
    )
    
    try:
        # If scraping all sorting methods
        if args.all:
            scraper.scrape_all_sorts(
                include_time_filters=args.include_time_filters,
                limit_per_request=args.limit_per_request,
                max_requests=args.max_requests,
                max_posts=args.max_posts,
                reset=args.reset
            )
        # If specified single sorting method
        elif args.sort:
            # Check if sorting method needs time filter
            if args.sort in ['top', 'controversial']:
                time_filter = args.time_filter or 'all'
            else:
                time_filter = None
                if args.time_filter:
                    logger.warning(f"Sorting method {args.sort} does not support time filter, --time-filter parameter will be ignored")
            
            scraper.scrape_by_sort(
                sort_method=args.sort,
                time_filter=time_filter,
                limit_per_request=args.limit_per_request,
                max_requests=args.max_requests,
                max_posts=args.max_posts,
                reset=args.reset
            )
        else:
            logger.error("Please specify sorting method (--sort) or use --all parameter to scrape all sorting methods")
            
    except KeyboardInterrupt:
        logger.info("User interrupted program.")
    except Exception as e:
        logger.error(f"Program execution error: {str(e)}")


if __name__ == "__main__":
    main() 