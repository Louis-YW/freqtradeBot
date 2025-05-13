import pandas as pd
import numpy as np
from pathlib import Path
from textblob import TextBlob
import logging
import re
from datetime import datetime
import json
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
import multiprocessing as mp
from functools import partial
import sys

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

# 初始化VADER分析器
vader = SentimentIntensityAnalyzer()

# 初始化FinBERT模型
try:
    finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    logger.info("FinBERT模型加载成功")
except Exception as e:
    logger.error(f"FinBERT模型加载失败: {str(e)}")
    finbert_tokenizer = None
    finbert_model = None

def get_sentiment_category(scores):
    """根据情感分数确定情感类别"""
    if scores['vader_compound'] > 0.05:
        return 'positive'
    elif scores['vader_compound'] < -0.05:
        return 'negative'
    else:
        return 'neutral'

def get_sentiment_intensity(scores):
    """计算情感强度"""
    intensity = abs(scores['vader_compound']) * (1 + scores['textblob_subjectivity'])
    return min(intensity, 1.0)  # 确保强度在0-1之间

def get_financial_sentiment(text):
    """使用FinBERT进行金融情感分析"""
    if not isinstance(text, str) or not finbert_model or not finbert_tokenizer:
        return {
            'financial_positive': 0,
            'financial_negative': 0,
            'financial_neutral': 0
        }
    
    try:
        # 截断文本以适应模型最大长度
        inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = finbert_model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        
        return {
            'financial_positive': probs[0][0].item(),
            'financial_negative': probs[0][1].item(),
            'financial_neutral': probs[0][2].item()
        }
    except Exception as e:
        logger.error(f"FinBERT分析失败: {str(e)}")
        return {
            'financial_positive': 0,
            'financial_negative': 0,
            'financial_neutral': 0
        }

def get_text_complexity(text):
    """计算文本复杂度"""
    if not isinstance(text, str):
        return 0
    try:
        # 分词
        tokens = word_tokenize(text.lower())
        # 去除停用词
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
        
        if not tokens:
            return 0
            
        # 计算平均词长
        avg_word_length = sum(len(word) for word in tokens) / len(tokens)
        # 计算词汇多样性（不同词的数量/总词数）
        vocabulary_diversity = len(set(tokens)) / len(tokens)
        
        return (avg_word_length * 0.5 + vocabulary_diversity * 0.5)
    except:
        return 0

def analyze_sentiment(text):
    """分析文本情绪"""
    if not isinstance(text, str):
        return {
            'textblob_polarity': 0,
            'textblob_subjectivity': 0,
            'vader_compound': 0,
            'vader_pos': 0,
            'vader_neg': 0,
            'vader_neu': 0,
            'text_complexity': 0,
            'sentiment_category': 'neutral',
            'sentiment_intensity': 0
        }
    try:
        # TextBlob分析
        textblob = TextBlob(text)
        
        # VADER分析
        vader_scores = vader.polarity_scores(text)
        
        # 文本复杂度
        complexity = get_text_complexity(text)
        
        # 金融情感分析
        financial_sentiment = get_financial_sentiment(text)
        
        # 情感分类
        sentiment_scores = {
            'vader_compound': vader_scores['compound'],
            'textblob_subjectivity': textblob.sentiment.subjectivity
        }
        sentiment_category = get_sentiment_category(sentiment_scores)
        
        # 情感强度
        sentiment_intensity = get_sentiment_intensity(sentiment_scores)
        
        return {
            'textblob_polarity': textblob.sentiment.polarity,
            'textblob_subjectivity': textblob.sentiment.subjectivity,
            'vader_compound': vader_scores['compound'],
            'vader_pos': vader_scores['pos'],
            'vader_neg': vader_scores['neg'],
            'vader_neu': vader_scores['neu'],
            'text_complexity': complexity,
            'sentiment_category': sentiment_category,
            'sentiment_intensity': sentiment_intensity,
            **financial_sentiment
        }
    except:
        return {
            'textblob_polarity': 0,
            'textblob_subjectivity': 0,
            'vader_compound': 0,
            'vader_pos': 0,
            'vader_neg': 0,
            'vader_neu': 0,
            'text_complexity': 0,
            'sentiment_category': 'neutral',
            'sentiment_intensity': 0,
            'financial_positive': 0,
            'financial_negative': 0,
            'financial_neutral': 0
        }

def clean_datetime(dt_str):
    """清理和标准化日期时间字符串"""
    if not isinstance(dt_str, str):
        return None
    try:
        # 移除 "Last edit: " 前缀和 " by xxx" 后缀
        dt_str = dt_str.replace("Last edit: ", "").split(" by ")[0]
        # 解析日期时间
        return pd.to_datetime(dt_str)
    except:
        return None

def process_sentiment_features(df, time_granularity='1H'):
    """
    处理情绪特征
    
    参数:
        df: 包含文本数据的DataFrame
        time_granularity: 时间粒度，默认为1小时
    """
    logger.info("\n==================================================")
    logger.info(f"开始处理 {time_granularity} 时间粒度的数据...")
    logger.info("==================================================\n")
    
    # 创建DataFrame的副本
    df = df.copy()
    
    # 1. 清理和转换时间列
    logger.info("1. 开始处理时间列...")
    df['created_time'] = df['created_time'].apply(clean_datetime)
    df = df.dropna(subset=['created_time'])
    
    if df.empty:
        logger.warning("清理后的数据为空，跳过处理")
        return
        
    logger.info(f"输入数据大小: {df.shape}")
    logger.info(f"时间范围: {df['created_time'].min()} 到 {df['created_time'].max()}")
    
    # 2. 处理情绪分数
    logger.info("2. 开始处理情绪分数...")
    
    # 3. 计算综合情绪分数
    logger.info("\n3. 计算综合情绪分数...")
    
    # 计算综合情绪分数
    df['sentiment_score'] = (
        df['textblob_polarity'] * 0.3 +
        df['vader_compound'] * 0.3 +
        df['financial_positive'] * 0.2 -
        df['financial_negative'] * 0.2
    )
    
    # 根据综合分数确定情感类别
    df['sentiment_category'] = df['sentiment_score'].apply(
        lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
    )
    
    # 4. 输出情绪分数统计
    logger.info("\n4. 情绪分数统计:")
    logger.info("-" * 30)
    
    for col in ['textblob_polarity', 'vader_compound', 'financial_positive', 'financial_negative', 'sentiment_score']:
        if col in df.columns:
            stats = df[col].describe()
            non_zero = (df[col] != 0).mean() * 100
            logger.info(f"{col}:")
            logger.info(f"  - 范围: {stats['min']:.3f} 到 {stats['max']:.3f}")
            logger.info(f"  - 均值: {stats['mean']:.3f}")
            logger.info(f"  - 标准差: {stats['std']:.3f}")
            logger.info(f"  - 非零值比例: {non_zero:.2f}%")
    
    # 5. 情感分类统计
    logger.info("\n5. 情感分类统计:")
    logger.info("-" * 30)
    
    sentiment_counts = df['sentiment_category'].value_counts()
    total = len(df)
    for category, count in sentiment_counts.items():
        percentage = (count / total) * 100
        logger.info(f"{category}: {count} ({percentage:.1f}%)")
    
    # 6. 按时间粒度聚合数据
    logger.info("\n6. 开始按时间粒度聚合数据...")
    logger.info(f"时间粒度: {time_granularity}")
    
    # 按时间粒度重采样
    resampled = df.set_index('created_time').resample(time_granularity).agg({
        'sentiment_score': 'mean',
        'sentiment_category': lambda x: x.mode().iloc[0] if not x.empty else 'neutral',
        'textblob_polarity': 'mean',
        'vader_compound': 'mean',
        'financial_positive': 'mean',
        'financial_negative': 'mean'
    })
    
    # 重置索引
    resampled = resampled.reset_index()
    
    # 7. 保存处理后的数据
    output_file = Path(__file__).parent.parent / 'data' / 'processed' / f'sentiment_features_{time_granularity}.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    resampled.to_csv(output_file, index=False)
    logger.info(f"\n处理完成！数据已保存到: {output_file}")
    
    return resampled

def load_and_prepare_data(test_mode=True):
    """
    加载和准备数据
    
    参数:
        test_mode: 是否使用测试模式（只处理少量数据）
    """
    # 1. 读取所有原始帖子
    bitcointalk_dir = Path(__file__).parent.parent / 'data/topics/bitcointalk'
    reddit_dir = Path(__file__).parent.parent / 'data/topics/reddit'
    posts_data = []
    comments_data = []
    
    # 检查目录是否存在
    if not bitcointalk_dir.exists():
        logger.warning(f"Bitcointalk目录不存在: {bitcointalk_dir}")
    if not reddit_dir.exists():
        logger.warning(f"Reddit目录不存在: {reddit_dir}")
    
    # 读取Bitcointalk数据
    bitcointalk_files = list(bitcointalk_dir.glob('*.json'))
    logger.info(f"找到 {len(bitcointalk_files)} 个Bitcointalk文件")
    
    total_posts = 0
    total_comments = 0
    
    for file in tqdm(bitcointalk_files, desc="读取文件"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 处理主帖
                post_data = {
                    'id': data['id'],
                    'title': data['title'],
                    'author': data['author'],
                    'created_time': data['created_time'],
                    'content': data['content'],
                    'score': data['score'],
                    'url': data['url'],
                    'type': 'post'
                }
                posts_data.append(post_data)
                total_posts += 1
                
                # 处理评论
                comments = data.get('comments', [])
                for comment in comments:
                    comment_data = {
                        'id': f"{data['id']}_comment",
                        'parent_id': data['id'],
                        'author': comment['author'],
                        'created_time': comment['created_time'],
                        'content': comment['body'],
                        'score': comment['score'],
                        'type': 'comment'
                    }
                    comments_data.append(comment_data)
                    total_comments += 1
                
            logger.info(f"成功读取文件: {file}")
            logger.info(f"  - 主帖ID: {data['id']}")
            logger.info(f"  - 评论数量: {len(comments)}")
        except Exception as e:
            logger.error(f"读取文件失败 {file}: {str(e)}")
    
    logger.info(f"总计处理了 {total_posts} 个主帖和 {total_comments} 条评论")
    
    # 转换为DataFrame
    posts_df = pd.DataFrame(posts_data)
    comments_df = pd.DataFrame(comments_data)
    
    logger.info(f"主帖DataFrame大小: {posts_df.shape}")
    logger.info(f"评论DataFrame大小: {comments_df.shape}")
    
    # 合并主帖和评论
    all_posts = pd.concat([posts_df, comments_df], ignore_index=True)
    logger.info(f"合并后的DataFrame大小: {all_posts.shape}")
    
    if test_mode:
        # 在测试模式下只返回前100条数据
        all_posts = all_posts.head(100)
        logger.info(f"测试模式：只使用前100条数据")
    
    # 添加情绪分析
    logger.info("开始进行情绪分析...")
    sentiment_results = []
    for text in tqdm(all_posts['content'], desc="分析情绪"):
        sentiment_results.append(analyze_sentiment(text))
    
    # 将情绪分析结果添加到DataFrame
    sentiment_df = pd.DataFrame(sentiment_results)
    all_posts = pd.concat([all_posts, sentiment_df], axis=1)
    
    return all_posts

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("\n开始数据处理流程...")
    logging.info("="*50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        test_mode = False
        logging.info("运行模式: 完整数据处理")
    else:
        test_mode = True
        logging.info("运行模式: 测试模式（仅处理少量数据）")
    
    # 检查各个时间粒度的数据文件
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    hourly_file = data_dir / 'sentiment_features_1H.csv'
    four_hour_file = data_dir / 'sentiment_features_4H.csv'
    daily_file = data_dir / 'sentiment_features_1D.csv'
    weekly_file = data_dir / 'sentiment_features_1W.csv'
    
    # 1. 检查是否需要情绪分析
    if hourly_file.exists():
        logging.info("发现已存在的1H数据文件，跳过情绪分析")
        df = None
    else:
        logging.info("开始加载数据并进行情绪分析...")
        df = load_and_prepare_data(test_mode=test_mode)
    
    # 2. 处理1H数据
    if df is not None:
        logging.info("\n处理1H时间粒度数据...")
        hourly_data = process_sentiment_features(df=df, time_granularity='1H')
    
    # 3. 处理4H数据
    if four_hour_file.exists():
        logging.info("\n发现已存在的4H数据文件，跳过处理")
    else:
        logging.info("\n处理4H时间粒度数据...")
        if hourly_file.exists():
            hourly_df = pd.read_csv(hourly_file)
            hourly_df['created_time'] = pd.to_datetime(hourly_df['created_time'])
            
            # 验证数据
            if hourly_df.empty:
                logging.error("1H数据为空，无法处理4H数据")
            elif not all(col in hourly_df.columns for col in ['sentiment_score', 'sentiment_category', 'textblob_polarity', 
                              'vader_compound', 'financial_positive', 'financial_negative']):
                logging.error("1H数据缺少必要的列，无法处理4H数据")
            else:
                # 按时间粒度重采样，处理空值
                resampled = hourly_df.set_index('created_time').resample('4H').agg({
                    'sentiment_score': lambda x: x.mean() if not x.empty and not x.isna().all() else None,
                    'sentiment_category': lambda x: x.mode().iloc[0] if not x.empty and not x.isna().all() else 'neutral',
                    'textblob_polarity': lambda x: x.mean() if not x.empty and not x.isna().all() else None,
                    'vader_compound': lambda x: x.mean() if not x.empty and not x.isna().all() else None,
                    'financial_positive': lambda x: x.mean() if not x.empty and not x.isna().all() else None,
                    'financial_negative': lambda x: x.mean() if not x.empty and not x.isna().all() else None
                })
                
                # 验证聚合结果
                if not resampled.empty:
                    # 检查是否有太多空值
                    null_percentage = resampled.isnull().mean() * 100
                    if (null_percentage > 50).any():
                        logging.warning(f"4H数据中存在大量空值: \n{null_percentage}")
                    
                    resampled = resampled.reset_index()
                    four_hour_file.parent.mkdir(parents=True, exist_ok=True)
                    resampled.to_csv(four_hour_file, index=False)
                    logging.info(f"4H数据已保存到: {four_hour_file}")
                    logging.info(f"4H数据大小: {resampled.shape}")
                else:
                    logging.error("4H数据聚合结果为空")
    
    # 4. 处理1D数据
    if daily_file.exists():
        logging.info("\n发现已存在的1D数据文件，跳过处理")
    else:
        logging.info("\n处理1D时间粒度数据...")
        if hourly_file.exists():
            hourly_df = pd.read_csv(hourly_file)
            hourly_df['created_time'] = pd.to_datetime(hourly_df['created_time'])
            
            # 验证数据
            if hourly_df.empty:
                logging.error("1H数据为空，无法处理1D数据")
            elif not all(col in hourly_df.columns for col in ['sentiment_score', 'sentiment_category', 'textblob_polarity', 
                              'vader_compound', 'financial_positive', 'financial_negative']):
                logging.error("1H数据缺少必要的列，无法处理1D数据")
            else:
                # 按时间粒度重采样，处理空值
                resampled = hourly_df.set_index('created_time').resample('1D').agg({
                    'sentiment_score': lambda x: x.mean() if not x.empty and not x.isna().all() else None,
                    'sentiment_category': lambda x: x.mode().iloc[0] if not x.empty and not x.isna().all() else 'neutral',
                    'textblob_polarity': lambda x: x.mean() if not x.empty and not x.isna().all() else None,
                    'vader_compound': lambda x: x.mean() if not x.empty and not x.isna().all() else None,
                    'financial_positive': lambda x: x.mean() if not x.empty and not x.isna().all() else None,
                    'financial_negative': lambda x: x.mean() if not x.empty and not x.isna().all() else None
                })
                
                # 验证聚合结果
                if not resampled.empty:
                    # 检查是否有太多空值
                    null_percentage = resampled.isnull().mean() * 100
                    if (null_percentage > 50).any():
                        logging.warning(f"1D数据中存在大量空值: \n{null_percentage}")
                    
                    resampled = resampled.reset_index()
                    daily_file.parent.mkdir(parents=True, exist_ok=True)
                    resampled.to_csv(daily_file, index=False)
                    logging.info(f"1D数据已保存到: {daily_file}")
                    logging.info(f"1D数据大小: {resampled.shape}")
                else:
                    logging.error("1D数据聚合结果为空")
    
    # 5. 处理1W数据
    if weekly_file.exists():
        logging.info("\n发现已存在的1W数据文件，跳过处理")
    else:
        logging.info("\n处理1W时间粒度数据...")
        if hourly_file.exists():
            hourly_df = pd.read_csv(hourly_file)
            hourly_df['created_time'] = pd.to_datetime(hourly_df['created_time'])
            
            # 验证数据
            if hourly_df.empty:
                logging.error("1H数据为空，无法处理1W数据")
            elif not all(col in hourly_df.columns for col in ['sentiment_score', 'sentiment_category', 'textblob_polarity', 
                              'vader_compound', 'financial_positive', 'financial_negative']):
                logging.error("1H数据缺少必要的列，无法处理1W数据")
            else:
                # 按时间粒度重采样，处理空值
                resampled = hourly_df.set_index('created_time').resample('1W').agg({
                    'sentiment_score': lambda x: x.mean() if not x.empty and not x.isna().all() else None,
                    'sentiment_category': lambda x: x.mode().iloc[0] if not x.empty and not x.isna().all() else 'neutral',
                    'textblob_polarity': lambda x: x.mean() if not x.empty and not x.isna().all() else None,
                    'vader_compound': lambda x: x.mean() if not x.empty and not x.isna().all() else None,
                    'financial_positive': lambda x: x.mean() if not x.empty and not x.isna().all() else None,
                    'financial_negative': lambda x: x.mean() if not x.empty and not x.isna().all() else None
                })
                
                # 验证聚合结果
                if not resampled.empty:
                    # 检查是否有太多空值
                    null_percentage = resampled.isnull().mean() * 100
                    if (null_percentage > 50).any():
                        logging.warning(f"1W数据中存在大量空值: \n{null_percentage}")
                    
                    resampled = resampled.reset_index()
                    weekly_file.parent.mkdir(parents=True, exist_ok=True)
                    resampled.to_csv(weekly_file, index=False)
                    logging.info(f"1W数据已保存到: {weekly_file}")
                    logging.info(f"1W数据大小: {resampled.shape}")
                else:
                    logging.error("1W数据聚合结果为空")
    
    logging.info("\n数据处理完成!")
    logging.info("="*50)
    if test_mode:
        logging.info("\n测试模式运行完成。如果以上输出看起来正常，可以使用 --full 参数运行完整数据处理：")
        logging.info("python process_sentiment_features.py --full")