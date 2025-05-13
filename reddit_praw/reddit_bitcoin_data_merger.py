#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reddit r/Bitcoin Data Merger Tool

This script is used to merge Reddit post data from multiple JSON files, remove duplicates, and sort them in chronological order.
It is suitable for processing data scraped with reddit_bitcoin_multi_sort_scraper.py in multiple sorting methods.
"""

import os
import json
import glob
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# 设置日志记录
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_merger.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_json_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSON file

    Args:
        filepath: Path to the JSON file

    Returns:
        A list containing post data
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} entries from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        return []


def merge_posts(data_dir: str, output_file: str = None) -> None:
    """
    Merge all post data in the directory, remove duplicates, and sort by time

    Args:
        data_dir: Data directory
        output_file: Output file path, automatically generated if None
    """
    # 获取所有JSON文件
    file_pattern = os.path.join(data_dir, "bitcoin_*.json")
    json_files = glob.glob(file_pattern)
    
    if not json_files:
        logger.error(f"No JSON files found in {data_dir} directory matching 'bitcoin_*.json' pattern")
        return
    
    logger.info(f"Found {len(json_files)} JSON files")
    
    # 合并数据
    all_posts = []
    for json_file in json_files:
        posts = load_json_data(json_file)
        all_posts.extend(posts)
    
    logger.info(f"Loaded a total of {len(all_posts)} post entries")
    
    # 去除重复项（使用ID作为唯一标识）
    unique_posts = {}
    for post in all_posts:
        post_id = post['id']
        if post_id not in unique_posts:
            unique_posts[post_id] = post
        else:
            # 如果已存在，保留更完整的记录（可选）
            # 例如，选择字段更多的记录
            if len(post) > len(unique_posts[post_id]):
                unique_posts[post_id] = post
    
    logger.info(f"{len(unique_posts)} post entries remaining after deduplication")
    
    # 将字典转换回列表
    merged_posts = list(unique_posts.values())
    
    # 按时间戳排序（从早到晚）
    merged_posts.sort(key=lambda x: x['created_utc'])
    
    logger.info(f"Sorted by timestamp from earliest to latest")
    
    # 生成输出文件名
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(data_dir, f"bitcoin_merged_{timestamp}.json")
    
    # 保存合并后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_posts, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved the merged {len(merged_posts)} post entries to {output_file}")
    
    # 输出排序统计信息
    sort_stats = {}
    for json_file in json_files:
        filename = os.path.basename(json_file)
        if filename.startswith("bitcoin_"):
            parts = filename.split("_")
            if len(parts) > 1:
                sort_type = parts[1]
                if sort_type not in sort_stats:
                    sort_stats[sort_type] = 0
                sort_stats[sort_type] += len(load_json_data(json_file))
    
    logger.info("Statistics of data by sorting method:")
    for sort_type, count in sort_stats.items():
        logger.info(f"  - {sort_type}: {count} entries")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Reddit r/Bitcoin Data Merger Tool")
    parser.add_argument("--data-dir", help="Data directory", default="data")
    parser.add_argument("--output", help="Output file path")
    args = parser.parse_args()
    
    # 确保数据目录存在
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory {args.data_dir} does not exist")
        return
    
    merge_posts(args.data_dir, args.output)


if __name__ == "__main__":
    main() 