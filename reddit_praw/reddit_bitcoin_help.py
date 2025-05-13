#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reddit r/Bitcoin Scraping Tool Usage Help

This script is used to display the usage instructions and parameter options for various scraping scripts in the project.
"""

import os
import argparse
import subprocess
import sys


def display_banner():
    """Display project banner"""
    banner = """
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃                                                                          ┃
    ┃   ██████╗ ███████╗██████╗ ██████╗ ██╗████████╗                          ┃
    ┃   ██╔══██╗██╔════╝██╔══██╗██╔══██╗██║╚══██╔══╝                          ┃
    ┃   ██████╔╝█████╗  ██║  ██║██║  ██║██║   ██║                             ┃
    ┃   ██╔══██╗██╔══╝  ██║  ██║██║  ██║██║   ██║                             ┃
    ┃   ██║  ██║███████╗██████╔╝██████╔╝██║   ██║                             ┃
    ┃   ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═════╝ ╚═╝   ╚═╝                             ┃
    ┃                                                                          ┃
    ┃   ██████╗ ██╗████████╗ ██████╗ ██████╗ ██╗███╗   ██╗                    ┃
    ┃   ██╔══██╗██║╚══██╔══╝██╔════╝██╔═══██╗██║████╗  ██║                    ┃
    ┃   ██████╔╝██║   ██║   ██║     ██║   ██║██║██╔██╗ ██║                    ┃
    ┃   ██╔══██╗██║   ██║   ██║     ██║   ██║██║██║╚██╗██║                    ┃
    ┃   ██████╔╝██║   ██║   ╚██████╗╚██████╔╝██║██║ ╚████║                    ┃
    ┃   ╚═════╝ ╚═╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝                    ┃
    ┃                                                                          ┃
    ┃   ███████╗ ██████╗██████╗  █████╗ ██████╗ ███████╗██████╗               ┃
    ┃   ██╔════╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗              ┃
    ┃   ███████╗██║     ██████╔╝███████║██████╔╝█████╗  ██████╔╝              ┃
    ┃   ╚════██║██║     ██╔══██╗██╔══██║██╔═══╝ ██╔══╝  ██╔══██╗              ┃
    ┃   ███████║╚██████╗██║  ██║██║  ██║██║     ███████╗██║  ██║              ┃
    ┃   ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝              ┃
    ┃                                                                          ┃
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """
    print(banner)


def check_environment():
    """Check environment configuration"""
    env_exists = os.path.exists(".env")
    example_exists = os.path.exists(".env.example")
    
    if env_exists:
        print("✓ .env configuration file detected")
    else:
        print("✗ .env configuration file not detected")
        if example_exists:
            print("  Hint: Please copy .env.example to .env and fill in your Reddit API credentials")
        else:
            print("  Hint: Please create a .env file and fill in your Reddit API credentials")
    
    # Check if dependencies are installed
    try:
        import praw
        import dotenv
        import tqdm
        print("✓ Required dependencies are installed")
    except ImportError as e:
        print(f"✗ Missing dependency: {str(e)}")
        print("  Hint: Please run 'pip install -r requirements.txt' to install the required dependencies")


def show_script_help(script_name):
    """Display help information for a specified script"""
    try:
        help_text = subprocess.check_output([sys.executable, script_name, "--help"], 
                                          stderr=subprocess.STDOUT,
                                          universal_newlines=True)
        print(f"\n{script_name} help information:")
        print("=" * 80)
        print(help_text)
    except subprocess.CalledProcessError as e:
        print(f"Error getting help for {script_name}: {str(e)}")
    except FileNotFoundError:
        print(f"Script file not found: {script_name}")


def show_usage_examples():
    """Show usage examples"""
    examples = """
Usage Examples:
=========

1. Basic Scraping
-------------
# Scrape r/Bitcoin posts with default parameters
python reddit_bitcoin_scraper.py

# Limit to a maximum of 1000 posts
python reddit_bitcoin_scraper.py --limit 1000

# Custom configuration file and output directory
python reddit_bitcoin_scraper.py --config myconfig.env --output bitcoin_data

2. Advanced Scraping
-------------
# Scrape r/Bitcoin posts with default parameters
python reddit_bitcoin_scraper_advanced.py

# Scrape hot posts, 50 posts per request
python reddit_bitcoin_scraper_advanced.py --sort hot --limit-per-request 50

# Limit to a maximum of 100 requests
python reddit_bitcoin_scraper_advanced.py --max-requests 100

3. Date Range Scraping
-------------
# Scrape posts from the last week
python reddit_bitcoin_scraper_by_date.py --days-ago 7

# Scrape posts from January 1, 2023 to June 30, 2023
python reddit_bitcoin_scraper_by_date.py --start-date 2023-01-01 --end-date 2023-06-30

# Scrape hot posts since January 1, 2023
python reddit_bitcoin_scraper_by_date.py --start-date 2023-01-01 --sort hot
"""
    print(examples)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Reddit r/Bitcoin Scraping Tool Usage Help")
    parser.add_argument("--script", choices=["basic", "advanced", "date", "all"], 
                      help="Display help information for a specified script")
    parser.add_argument("--examples", action="store_true", help="Display usage examples")
    parser.add_argument("--check", action="store_true", help="Check environment configuration")
    args = parser.parse_args()
    
    display_banner()
    print("\nReddit r/Bitcoin Posts Scraping Tool\n")
    
    # If no arguments are provided, display all information
    if len(sys.argv) == 1:
        check_environment()
        show_usage_examples()
        return
    
    # Check environment
    if args.check:
        check_environment()
    
    # Display script help
    if args.script:
        if args.script == "basic" or args.script == "all":
            show_script_help("reddit_bitcoin_scraper.py")
        
        if args.script == "advanced" or args.script == "all":
            show_script_help("reddit_bitcoin_scraper_advanced.py")
        
        if args.script == "date" or args.script == "all":
            show_script_help("reddit_bitcoin_scraper_by_date.py")
    
    # Display usage examples
    if args.examples:
        show_usage_examples()


if __name__ == "__main__":
    main() 