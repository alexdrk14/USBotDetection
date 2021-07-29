from utils import *
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, roc_curve, auc, precision_recall_curve, accuracy_score, average_precision_score
from sklearn.metrics import plot_confusion_matrix


#from DataScaler import DataScaler

import numpy as np
from collections import Counter
from collections import defaultdict
import argparse, sys
from datetime import datetime
#from model_params import params
import random

#Data scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

#oversampling
from imblearn.combine import SMOTETomek
from sklearn.utils import shuffle

objective = "multi:softprob"
learning_rate = 0.078
xgb_n_estimators = 490
max_depth = 4
colsample_bytree = 0.5
eval_metric = "aucpr"
"""
#freq_features_list = ['mst_fr_ment_rt_word_1_6', 'geo', 'mst_fr_hs_rt_word_3_6', 'mst_fr_hs_rt_word_3_7',
# 'mst_fr_hs_tw_word_2_2', 'mst_fr_ment_tw_3', 'followers', 'mst_fr_ment_rt_word_2_6', 'mst_fr_ment_rt_word_3_6',
# 'location', 'tw_urls_std', 'rt_time_max', 'mst_fr_hs_tw_word_3_2', 'mst_fr_hs_tw_word_2_9', 'mst_fr_ment_rt_word_1_5',
# 'mst_fr_hs_tw_word_1_2', 'mst_fr_ment_tw_word_1_2', 'mst_fr_ment_tw_word_2_2', 'mst_fr_ment_rt_word_1_2', 'mst_fr_hs_rt_word_2_7', 'mst_fr_hs_rt_word_1_7', 'mst_fr_ment_rt_word_1_7', 'in_degree', 'mst_fr_hs_rt_word_1_2', 'mst_fr_hs_rt_word_3_9', 'mst_fr_ment_rt_word_3_9', 'mst_fr_hs_rt_word_1_3', 'mst_fr_hs_tw_word_1_9', 'default_prof', 'listed', 'mst_fr_ment_rt_word_3_4', 'mst_fr_hs_rt_word_1_4', 'entities', 'daily_rt_tw_6', 'mst_fr_ment_tw_word_1_7', 'friends_count', 'mst_fr_ment_rt_3', 'rt_avg', 'mst_fr_ment_tw_word_1_0', 'description', 'mst_fr_ment_rt_word_2_2', 'mst_fr_hs_rt_word_1_9', 'mst_fr_hs_rt_word_2_2', 'mst_fr_ment_rt_word_1_4', 'mst_fr_ment_rt_word_1_9', 'mst_fr_hs_rt_word_2_6', 'tw_urls_avg', 'mst_fr_ment_tw_2', 'mst_fr_hs_rt_word_2_9', 'mst_fr_hs_rt_word_2_4', 'mst_fr_hs_rt_word_1_0', 'mst_fr_hs_rt_word_1_6', 'mst_fr_ment_tw_word_3_2', 'mst_fr_hs_rt_word_1_1', 'mst_fr_hs_rt_word_2_3', 'mst_fr_hs_tw_word_1_7', 'mst_fr_hs_rt_word_1_8', 'tw_ment_avg', 'mst_fr_ment_rt_word_3_2', 'mst_fr_hs_tw_word_3_9', 'mst_fr_hs_rt_word_3_2', 'daily_rt_tw_1', 'mst_fr_hs_rt_word_1_5', 'mst_fr_ment_rt_word_2_9', 'mst_fr_ment_tw_word_3_7', 'daily_rt_tw_3', 'favourites', 'rt_time_avg', 'mst_fr_hs_tw_word_2_7', 'hour_rt_tw_13', 'rt_urls_avg', 'rt_urls_std', 'hour_rt_tw_1', 'description_len', 'daily_rt_tw_0', 'mst_fr_hs_rt_1', 'hour_rt_tw_11', 'daily_rt_6', 'rt_time_std', 'mst_fr_hs_rt_word_2_8', 'mst_fr_hs_tw_word_3_7', 'daily_rt_tw_2', 'mst_fr_ment_rt_2', 'hour_rt_tw_12', 'mst_fr_ment_rt_word_1_0', 'daily_rt_0', 'mst_fr_ment_tw_word_2_7', 'mst_fr_ment_rt_word_1_3', 'mst_fr_hs_tw_word_1_3', 'mst_fr_ment_rt_word_2_7', 'hour_rt_tw_20', 'hour_rt_20', 'hour_rt_12', 'mst_fr_hs_rt_word_2_0', 'hour_rt_tw_6', 'hour_rt_tw_23', 'tw_ment_std', 'rt_ment_std', 'hour_rt_21', 'mst_fr_hs_rt_2', 'mst_fr_ment_rt_word_1_8', 'daily_rt_5', 'hour_rt_1', 'mst_fr_ment_tw_1', 'mst_fr_ment_rt_word_3_7', 'hour_rt_tw_9', 'mst_fr_hs_rt_word_2_1', 'words_frq_rt_1_5', 'hour_rt_23', 'hour_rt_13', 'mst_fr_hs_tw_word_3_6', 'mst_fr_ment_tw_word_1_6', 'mst_fr_hs_tw_word_3_3', 'hour_rt_6', 'hour_rt_tw_14', 'daily_rt_tw_4', 'mst_fr_hs_tw_word_1_4', 'mst_fr_ment_rt_word_2_3', 'rt_time_min', 'mst_fr_ment_rt_word_3_0', 'rt_hash_std', 'words_frq_rt_2_9', 'w_out_degree', 'hour_rt_tw_22', 'words_frq_rt_1_1', 'mst_fr_hs_rt_word_3_3', 'words_frq_rt_1_7', 'mst_fr_hs_tw_word_1_5', 'daily_rt_2', 'words_frq_rt_2_6', 'hour_rt_2', 'hour_rt_22', 'hour_rt_tw_21', 'rt_ment_avg', 'mst_fr_hs_rt_word_3_0', 'mst_fr_hs_rt_word_2_5', 'words_frq_rt_1_6', 'words_frq_rt_1_9', 'hour_rt_9', 'mst_fr_ment_rt_word_2_0', 'mst_fr_ment_rt_word_3_1', 'mst_fr_hs_rt_word_3_1', 'words_frq_rt_3_9', 'verified', 'hour_rt_10', 'hour_rt_tw_2', 'mst_fr_hs_tw_word_2_5', 'words_frq_rt_2_8', 'w_in_degree', 'mst_fr_ment_rt_word_3_5', 'daily_rt_tw_5', 'daily_rt_3', 'mst_fr_ment_tw_word_2_1', 'words_frq_tw_1_9', 'words_frq_rt_3_4', 'mst_fr_ment_rt_word_1_1', 'words_frq_rt_3_5', 'mst_fr_hs_tw_word_1_1', 'mst_fr_ment_tw_word_3_4', 'mst_fr_ment_tw_word_3_1', 'mst_fr_hs_tw_word_1_8', 'hour_rt_17', 'hour_rt_0', 'hour_rt_4', 'hour_rt_3', 'mst_fr_hs_rt_word_3_4', 'words_frq_rt_3_6', 'words_frq_tw_1_2', 'rt_hash_avg', 'hour_rt_14', 'words_frq_rt_2_1', 'hour_rt_tw_16', 'hour_rt_tw_8', 'daily_rt_1', 'hour_rt_15', 'mst_fr_hs_tw_word_1_6', 'hour_rt_tw_19', 'hour_rt_5', 'mst_fr_ment_rt_word_2_8', 'mst_fr_ment_tw_word_1_3', 'mst_fr_ment_rt_word_3_3', 'mst_fr_hs_tw_word_2_6', 'words_frq_rt_2_0', 'tw_avg', 'mst_fr_ment_tw_word_2_5', 'words_frq_rt_3_8', 'words_frq_tw_3_0', 'words_frq_rt_1_0', 'hour_rt_tw_7', 'hour_rt_16', 'tw_hash_avg', 'hour_tw_20', 'hour_rt_tw_17', 'words_frq_rt_2_3', 'words_frq_rt_1_3', 'hour_rt_11', 'statuses', 'mst_fr_ment_rt_word_2_4', 'w_degree', 'mst_fr_ment_tw_word_2_0', 'words_frq_rt_1_2', 'words_frq_tw_2_2', 'words_frq_rt_2_7', 'mst_fr_ment_rt_1', 'mst_fr_ment_rt_word_3_8', 'hour_tw_22', 'words_frq_tw_2_8', 'hour_rt_tw_15', 'hour_rt_tw_0', 'words_frq_rt_3_1', 'mst_fr_ment_tw_word_3_3', 'mst_fr_hs_tw_word_3_4', 'mst_fr_ment_tw_word_1_8', 'hour_rt_tw_5', 'hour_rt_8', 'mst_fr_ment_rt_word_2_5', 'name_screen_sim', 'hour_rt_18', 'mst_fr_hs_tw_word_2_8', 'mst_fr_hs_tw_word_2_1', 'hour_tw_1', 'words_frq_tw_1_8', 'words_frq_tw_3_9', 'mst_fr_ment_tw_word_2_3', 'hour_rt_tw_18', 'words_frq_tw_3_4', 'mst_fr_ment_rt_word_2_1', 'mst_fr_hs_tw_word_1_0', 'hour_tw_14', 'hour_tw_7', 'words_frq_rt_3_0', 'daily_tw_4', 'mst_fr_hs_tw_word_3_1', 'mst_fr_hs_rt_word_3_8', 'words_frq_rt_2_5', 'mst_fr_ment_tw_word_1_4', 'words_frq_tw_2_4', 'daily_rt_4', 'hour_tw_23', 'tw_hash_std', 'mst_fr_ment_tw_word_2_9', 'words_frq_rt_2_4', 'hour_rt_tw_4', 'mst_fr_hs_tw_word_2_3', 'mst_fr_ment_tw_word_1_1', 'mst_fr_ment_tw_word_3_0', 'words_frq_tw_2_0', 'mst_fr_ment_tw_word_3_8', 'words_frq_rt_3_7', 'mst_fr_ment_tw_word_1_9', 'words_frq_tw_3_2', 'mst_fr_hs_rt_3', 'hour_rt_19', 'mst_fr_hs_tw_word_3_8', 'mst_fr_hs_tw_1', 'mst_fr_hs_tw_word_2_0', 'hour_tw_6', 'words_frq_tw_3_6', 'hour_tw_8', 'hour_tw_15', 'mst_fr_ment_tw_word_2_8', 'words_frq_rt_3_2', 'mst_fr_hs_tw_word_2_4', 'hour_rt_tw_3', 'words_frq_rt_1_8', 'words_frq_tw_1_0', 'words_frq_tw_2_9', 'words_frq_rt_1_4', 'daily_tw_5', 'hour_rt_7', 'mst_fr_hs_tw_3', 'mst_fr_ment_tw_word_3_5', 'tw_rt_ration', 'words_frq_tw_3_7', 'words_frq_tw_2_3', 'hour_tw_3', 'hour_tw_13', 'daily_tw_0', 'words_frq_tw_3_5', 'hour_tw_16', 'mst_fr_ment_tw_word_2_6', 'mst_fr_ment_tw_word_3_6', 'out_degree', 'words_frq_rt_2_2', 'hour_tw_21', 'daily_tw_6', 'mst_fr_hs_tw_word_3_0', 'mst_fr_hs_tw_word_3_5', 'hour_tw_18', 'daily_tw_1', 'hour_rt_tw_10', 'mst_fr_ment_tw_word_3_9', 'hour_tw_12', 'daily_tw_3', 'words_frq_tw_1_1', 'mst_fr_hs_rt_word_3_5', 'hour_tw_9', 'words_frq_tw_1_4', 'words_frq_tw_2_5', 'hour_tw_17', 'mst_fr_ment_tw_word_1_5', 'words_frq_tw_1_7', 'words_frq_tw_2_6', 'words_frq_tw_1_3', 'words_frq_tw_3_8', 'daily_tw_2', 'hour_tw_0', 'words_frq_rt_3_3', 'words_frq_tw_2_1', 'hour_tw_2', 'rt_self', 'mst_fr_hs_tw_2', 'bckg_img', 'hour_tw_11', 'mst_fr_ment_tw_word_2_4', 'words_frq_tw_1_5', 'words_frq_tw_1_6', 'words_frq_tw_2_7', 'words_frq_tw_3_3', 'hour_tw_4', 'hour_tw_19', 'hour_tw_5', 'hour_tw_10']
freq_features_list = ['daily_rt_tw_5', 'N3_retweet_mentioned_word_7', 'N1_retweet_word_0', 'hour_rt_tw_23', 'daily_rt_tw_6', 'retweet_time_min',
                      'listed_by_age', 'N3_retweet_hastag_word_6', 'hour_rt_tw_3', 'daily_rt_tw_1', 'daily_tweet_avg', 'N3_retweet_hastag_word_5',
                      'daily_rt_6', 'N2_retweet_mentioned_tfidf', 'N1_tweet_mentioned_word_4', 'N3_tweet_mentioned_tfidf', 'N3_tweet_hastag_word_3',
                      'N1_tweet_hastag_word_1', 'N2_retweet_mentioned_word_7', 'N1_tweet_mentioned_word_0', 'statuses_count', 'tweet_freq_by_age',
                      'daily_rt_tw_3', 'N1_retweet_hastag_word_9', 'daily_retweet_avg', 'N2_tweet_word_3', 'N2_tweet_hastag_word_3', 'daily_rt_tw_2',
                      'N2_retweet_hastag_tfidf', 'hour_rt_22', 'N1_tweet_hastag_word_8', 'N2_retweet_mentioned_word_1', 'tweet_number_of_hashtags_std',
                      'N1_retweet_hastag_word_8', 'tweet_number_of_mentions_std', 'daily_rt_tw_0', 'N3_retweet_mentioned_word_1',
                      'N3_retweet_mentioned_word_4', 'N2_retweet_mentioned_word_4', 'retweet_number_of_urls_avg', 'N3_retweet_mentioned_tfidf',
                      'N1_tweet_hastag_word_6', 'N3_retweet_hastag_word_3', 'N1_tweet_mentioned_word_5', 'N2_retweet_mentioned_word_5',
                      'N2_tweet_mentioned_tfidf', 'N3_retweet_hastag_word_7', 'N3_retweet_mentioned_word_5', 'friends_by_age', 'N1_tweet_hastag_word_9',
                      'N2_tweet_mentioned_word_1', 'N1_retweet_mentioned_word_4', 'N1_tweet_hastag_word_4', 'N1_retweet_hastag_word_4', 'friends_count',
                      'N1_retweet_hastag_word_5', 'hour_rt_tw_12', 'hour_rt_tw_13', 'N1_tweet_mentioned_tfidf', 'N2_retweet_hastag_word_3',
                      'N2_retweet_mentioned_word_3', 'N1_retweet_mentioned_word_6', 'N2_retweet_hastag_word_5', 'N3_retweet_hastag_tfidf',
                      'N3_retweet_hastag_word_4', 'N1_retweet_hastag_word_7', 'daily_rt_2', 'foll_friends_rel', 'N1_retweet_hastag_word_3',
                      'description', 'favourites_count', 'N2_tweet_hastag_word_5', 'N1_tweet_hastag_word_3', 'N1_retweet_hastag_tfidf',
                      'N1_tweet_hastag_word_7', 'N2_retweet_hastag_word_7', 'entities_count', 'retweet_number_of_mentions_avg',
                      'followers_by_age', 'N1_tweet_hastag_word_0', 'N2_retweet_hastag_word_6', 'N2_retweet_hastag_word_0', 'geolocation',
                      'N2_tweet_mentioned_word_4', 'tweet_number_of_hashtags_avg', 'tweet_number_of_urls_std', 'N1_tweet_mentioned_word_1',
                      'N3_tweet_mentioned_word_4', 'N1_retweet_mentioned_word_3', 'N3_retweet_hastag_word_8', 'daily_rt_tw_4',
                      'N1_tweet_mentioned_word_6', 'N1_retweet_hastag_word_6', 'N2_retweet_mentioned_word_0', 'N1_retweet_mentioned_word_5',
                      'N1_retweet_mentioned_word_1', 'description_len', 'name_and_screen_name_similarity', 'followers_count',
                      'N1_retweet_hastag_word_1', 'retweet_number_of_urls_std', 'location', 'retweet_number_of_mentions_std', 'favourites_by_age',
                      'tweet_number_of_mentions_avg', 'listed_count', 'N2_retweet_hastag_word_4', 'daily_rt_0', 'N3_retweet_mentioned_word_3',
                      'N1_tweet_hastag_word_5', 'N2_retweet_hastag_word_8', 'N1_tweet_mentioned_word_7', 'N1_retweet_mentioned_word_7',
                      'N3_retweet_mentioned_word_0', 'hour_rt_13', 'hour_rt_23', 'N1_tweet_word_3', 'w_degree', 'hour_rt_tw_14', 'daily_rt_4',
                      'retweet_time_avg', 'daily_tw_2', 'N2_retweet_word_7', 'N1_retweet_mentioned_tfidf', 'hour_rt_14', 'tweet_number_of_urls_avg',
                      'hour_rt_1', 'N2_tweet_mentioned_word_7', 'N1_retweet_hastag_word_0', 'N3_retweet_hastag_word_1', 'hour_rt_21',
                      'N1_retweet_mentioned_word_0', 'N3_tweet_hastag_word_5', 'N1_retweet_mentioned_word_8', 'N2_tweet_hastag_word_6',
                      'N3_tweet_hastag_word_7', 'N2_retweet_hastag_word_1', 'hour_rt_3', 'hour_rt_2', 'hour_rt_17', 'N2_retweet_word_3',
                      'N3_retweet_mentioned_word_6', 'N1_retweet_hastag_word_2', 'N2_retweet_mentioned_word_6', 'N1_tweet_hastag_word_2',
                      'w_out_degree', 'hour_rt_20', 'N1_tweet_mentioned_word_8', 'N2_retweet_mentioned_word_8', 'hour_rt_tw_15',
                      'N3_tweet_mentioned_word_7', 'daily_rt_5', 'N2_retweet_hastag_word_2', 'daily_rt_1', 'hour_rt_tw_22', 'daily_rt_3',
                      'N3_tweet_mentioned_word_5', 'N2_tweet_hastag_word_7', 'hour_rt_12', 'hour_rt_9', 'hour_rt_tw_11', 'in_degree',
                      'hour_rt_tw_1', 'N3_retweet_hastag_word_2', 'N2_tweet_hastag_word_4', 'N1_tweet_mentioned_word_3', 'N3_tweet_word_3',
                      'hour_rt_11', 'N3_tweet_hastag_word_8', 'N2_retweet_word_0', 'hour_rt_tw_17', 'hour_rt_15', 'N2_retweet_hastag_word_9',
                      'N1_retweet_word_8', 'tweet_retweet_ration', 'N3_tweet_hastag_word_1', 'N3_tweet_mentioned_word_0', 'N3_tweet_mentioned_word_8',
                      'N2_tweet_hastag_word_0', 'N3_tweet_mentioned_word_1', 'N2_tweet_mentioned_word_8', 'N3_tweet_hastag_word_4', 'N2_retweet_word_5',
                      'hour_rt_tw_4', 'N2_tweet_mentioned_word_0', 'hour_rt_tw_18', 'retweet_number_of_hashtags_avg', 'hour_rt_4', 'hour_rt_0',
                      'N3_tweet_hastag_word_2', 'hour_rt_tw_6', 'hour_rt_tw_0', 'hour_rt_18', 'N3_tweet_hastag_word_0', 'hour_rt_tw_21',
                      'hour_rt_tw_9', 'hour_rt_6', 'N2_retweet_mentioned_word_9', 'hour_rt_tw_20', 'N3_retweet_hastag_word_9', 'rt_self',
                      'N2_retweet_word_4', 'N2_tweet_hastag_word_8', 'N2_tweet_hastag_tfidf', 'retweet_time_std', 'N2_tweet_hastag_word_9',
                      'N1_retweet_mentioned_word_2', 'w_in_degree', 'hour_rt_tw_2', 'hour_rt_tw_7', 'hour_rt_7', 'hour_rt_19', 'N1_tweet_word_5',
                      'hour_rt_tw_10', 'hour_tw_10', 'N3_tweet_word_5', 'N1_retweet_word_5', 'out_degree', 'hour_rt_8', 'N1_tweet_word_1',
                      'N3_tweet_mentioned_word_3', 'N1_tweet_mentioned_word_2', 'N1_retweet_mentioned_word_9', 'hour_rt_10', 'N2_tweet_mentioned_word_5',
                      'N1_retweet_word_9', 'N3_retweet_mentioned_word_9', 'N3_tweet_word_1', 'N1_retweet_word_6', 'N2_tweet_mentioned_word_6',
                      'N2_tweet_word_5', 'N1_retweet_word_3', 'hour_tw_22', 'hour_tw_20', 'N2_tweet_word_8', 'N2_tweet_hastag_word_1',
                      'hour_rt_tw_16', 'N3_tweet_word_9', 'N2_tweet_word_1', 'hour_tw_16', 'N1_tweet_hastag_tfidf', 'hour_tw_13',
                      'N3_retweet_mentioned_word_8', 'hour_tw_3', 'daily_tw_0', 'N3_tweet_hastag_word_9', 'N2_retweet_word_2',
                      'N1_retweet_word_4', 'hour_rt_5', 'N2_retweet_mentioned_word_2', 'hour_rt_16', 'hour_tw_21', 'N3_retweet_hastag_word_0',
                      'N3_retweet_word_3', 'N3_tweet_hastag_tfidf', 'N3_tweet_word_7', 'hour_tw_8', 'hour_tw_12', 'hour_rt_tw_19', 'hour_rt_tw_8',
                      'retweet_time_max', 'N1_retweet_word_7', 'N1_retweet_word_1', 'N3_retweet_word_1', 'N2_retweet_word_9', 'N2_retweet_word_6',
                      'N1_tweet_mentioned_word_9', 'daily_tw_1', 'N3_retweet_word_8', 'N2_tweet_word_7', 'N3_tweet_mentioned_word_2', 'hour_tw_14',
                      'hour_tw_17', 'daily_tw_5', 'default_profile', 'hour_tw_6', 'hour_tw_4', 'hour_tw_9', 'N3_retweet_mentioned_word_2',
                      'hour_tw_7', 'hour_tw_19', 'N2_tweet_hastag_word_2', 'N3_retweet_word_0', 'N3_tweet_hastag_word_6', 'daily_tw_3',
                      'N3_retweet_word_9', 'N3_tweet_mentioned_word_6', 'N2_tweet_word_0', 'N3_retweet_word_6', 'daily_tw_4', 'N2_tweet_word_2',
                      'hour_tw_2', 'hour_tw_5', 'background_img', 'name_length', 'N1_tweet_word_7', 'N3_tweet_mentioned_word_9', 'name_digits',
                      'daily_tw_6', 'N2_tweet_mentioned_word_3', 'N2_tweet_word_9', 'N1_tweet_word_6', 'retweet_number_of_hashtags_std', 'hour_tw_15',
                      'N2_retweet_word_1', 'N3_retweet_word_4', 'N2_retweet_word_8', 'N1_tweet_word_8', 'N1_tweet_word_4', 'hour_tw_23',
                      'N2_tweet_mentioned_word_9', 'N3_retweet_word_7', 'N1_tweet_word_0', 'N2_tweet_mentioned_word_2', 'N1_retweet_word_2',
                      'hour_tw_0', 'verified', 'hour_rt_tw_5', 'N1_tweet_word_9', 'N3_tweet_word_4', 'hour_tw_18', 'N3_retweet_word_5',
                      'N1_tweet_word_2', 'N2_tweet_word_6']

freq_features_list = ['N1_retweet_hastag_word_8', 'daily_retweet_avg', 'N1_tweet_hastag_word_9', 'entities_count', 'daily_rt_0',
                      'daily_rt_2', 'listed_count', 'N1_tweet_mentioned_tfidf', 'N1_tweet_hastag_word_8', 'N3_tweet_mentioned_tfidf',
                      'N1_retweet_hastag_word_9', 'retweet_number_of_urls_std', 'daily_rt_6', 'retweet_number_of_urls_avg',
                      'N1_tweet_hastag_word_5', 'N2_retweet_hastag_word_7', 'N1_retweet_hastag_word_3', 'tweet_number_of_hashtags_std',
                      'hour_rt_tw_13', 'daily_rt_tw_0', 'tweet_number_of_mentions_avg', 'daily_rt_tw_3', 'N1_tweet_hastag_word_3',
                      'N2_tweet_hastag_word_3', 'daily_rt_tw_5', 'description', 'N1_retweet_mentioned_word_1', 'tweet_number_of_mentions_std',
                      'N3_tweet_hastag_word_3', 'friends_by_age', 'N2_retweet_hastag_word_5', 'N3_retweet_hastag_word_3', 'N1_retweet_mentioned_word_4',
                      'geolocation', 'N1_tweet_mentioned_word_1', 'N1_retweet_mentioned_word_7', 'retweet_number_of_mentions_avg', 'N1_retweet_hastag_word_5',
                      'N2_retweet_hastag_word_3', 'location', 'N1_retweet_mentioned_word_3', 'N1_tweet_mentioned_word_4', 'N2_retweet_mentioned_word_5',
                      'N1_retweet_hastag_word_1', 'N1_retweet_hastag_word_6', 'tweet_number_of_urls_std', 'daily_tweet_avg', 'N1_retweet_mentioned_word_5',
                      'N2_retweet_mentioned_word_7', 'N3_retweet_hastag_word_5', 'N1_tweet_hastag_word_7', 'description_len',
                      'N3_retweet_mentioned_word_4', 'daily_rt_tw_4', 'N1_tweet_hastag_word_6', 'friends_count',
                      'N3_tweet_mentioned_word_4', 'N1_retweet_hastag_word_7', 'N2_tweet_mentioned_tfidf', 'daily_rt_tw_6',
                      'N2_tweet_mentioned_word_4', 'listed_by_age', 'N1_tweet_mentioned_word_0', 'N3_retweet_mentioned_tfidf',
                      'N1_retweet_hastag_word_4', 'daily_rt_tw_2', 'N1_tweet_mentioned_word_7', 'N3_retweet_hastag_word_8',
                      'tweet_number_of_hashtags_avg', 'N3_retweet_mentioned_word_1', 'N2_retweet_mentioned_word_1',
                      'daily_rt_tw_1', 'N2_tweet_mentioned_word_7', 'N2_retweet_mentioned_word_4', 'hour_rt_tw_12',
                      'N1_tweet_hastag_word_4', 'N3_retweet_mentioned_word_5', 'N1_retweet_hastag_tfidf', 'tweet_number_of_urls_avg',
                      'hour_rt_tw_3', 'N3_retweet_hastag_word_7', 'N1_tweet_hastag_word_0', 'N1_tweet_hastag_word_1', 'hour_rt_tw_23',
                      'retweet_number_of_mentions_std', 'daily_rt_4', 'N1_tweet_mentioned_word_5', 'N2_tweet_hastag_word_5', 'favourites_count',
                      'retweet_time_min', 'N3_retweet_mentioned_word_7', 'followers_count', 'N2_retweet_hastag_word_6', 'N1_retweet_word_0',
                      'hour_rt_13', 'N1_retweet_hastag_word_0', 'hour_rt_22', 'N3_retweet_hastag_word_6', 'N2_retweet_hastag_word_4',
                      'N2_retweet_mentioned_tfidf', 'N2_tweet_mentioned_word_1', 'hour_rt_tw_14', 'N3_retweet_mentioned_word_3',
                      'N3_retweet_hastag_tfidf', 'N3_retweet_hastag_word_1', 'N2_retweet_mentioned_word_3', 'followers_by_age',
                      'daily_rt_3', 'N2_retweet_hastag_word_8', 'hour_rt_23', 'daily_tw_2', 'N2_tweet_hastag_word_6', 'foll_friends_rel',
                      'N1_retweet_hastag_word_2', 'tweet_freq_by_age', 'daily_rt_1', 'N3_tweet_mentioned_word_5', 'hour_rt_0',
                      'hour_rt_tw_20', 'N1_tweet_mentioned_word_8', 'hour_rt_tw_4', 'N2_retweet_mentioned_word_0', 'N3_tweet_mentioned_word_7',
                      'rt_self', 'hour_rt_12', 'N3_tweet_mentioned_word_1', 'daily_rt_5', 'N1_tweet_hastag_word_2', 'hour_rt_tw_15',
                      'N2_tweet_mentioned_word_5', 'N1_retweet_mentioned_word_6', 'hour_rt_1', 'hour_rt_tw_0', 'hour_rt_tw_22',
                      'N2_tweet_hastag_tfidf', 'hour_rt_3', 'N2_tweet_hastag_word_7', 'hour_rt_20', 'N3_tweet_hastag_word_1',
                      'hour_rt_2', 'hour_rt_tw_11', 'hour_rt_tw_18', 'N3_retweet_hastag_word_4', 'N3_retweet_mentioned_word_0',
                      'N2_tweet_mentioned_word_8', 'hour_rt_14', 'hour_rt_tw_2', 'hour_rt_tw_1', 'hour_rt_11', 'hour_tw_19',
                      'N2_retweet_hastag_word_0', 'N3_tweet_hastag_word_5', 'N2_retweet_mentioned_word_8', 'N2_tweet_word_3',
                      'tweet_retweet_ration', 'hour_tw_11', 'hour_rt_10', 'hour_tw_8', 'N2_tweet_hastag_word_9']


freq_features_list = ['hour_rt_9','entities_count','N1_tweet_hastag_word_6','friends_by_age','tweet_number_of_mentions_avg',
 'description','N1_retweet_hastag_word_0','N3_retweet_mentioned_word_3','daily_rt_tw_6','daily_rt_tw_4','N3_retweet_mentioned_tfidf',
 'N3_retweet_hastag_word_5','N2_retweet_mentioned_word_3','N1_retweet_hastag_word_3','N3_retweet_mentioned_word_7','daily_rt_tw_1','N3_tweet_mentioned_tfidf',
 'N2_tweet_hastag_word_6','retweet_time_std','N3_retweet_mentioned_word_1','N1_tweet_mentioned_word_4','N1_tweet_word_3',
 'N2_tweet_hastag_word_5','N1_tweet_hastag_word_3','N1_retweet_hastag_word_8','N2_tweet_hastag_word_7','retweet_time_max',
 'hour_rt_tw_13','tweet_number_of_mentions_std','N2_retweet_mentioned_word_1','followers_by_age','tweet_number_of_urls_avg','favourites_count',
 'N1_retweet_mentioned_word_5','geolocation','N2_retweet_hastag_word_3','N2_retweet_mentioned_word_7','tweet_by_age',
 'N2_tweet_hastag_word_3','N1_retweet_hastag_word_6','N3_retweet_hastag_word_7','N3_retweet_hastag_word_3','location',
 'N2_retweet_mentioned_word_4','N3_retweet_mentioned_word_5','N3_retweet_hastag_tfidf','listed_by_age',
 'friends_count','N1_tweet_mentioned_word_0','N1_tweet_hastag_word_7','N2_retweet_hastag_word_7','followers_count',
 'N1_retweet_mentioned_word_7','description_length','hour_rt_22','default_profile','N1_tweet_hastag_word_5',
 'N3_retweet_hastag_word_6','daily_rt_tw_3','N2_retweet_hastag_word_6','retweet_time_avg','N1_retweet_hastag_word_5',
 'N2_retweet_hastag_word_8','N1_retweet_mentioned_word_1','N3_tweet_hastag_word_5','N1_retweet_mentioned_word_6','N1_retweet_hastag_word_7',
 'N2_retweet_mentioned_word_5','N2_retweet_hastag_word_5','listed_count','N2_retweet_mentioned_word_0',
 'N1_tweet_mentioned_word_1','N1_retweet_mentioned_word_3','N1_retweet_mentioned_word_4','daily_rt_0',
 'N1_retweet_hastag_tfidf','N1_tweet_hastag_word_8','N3_tweet_mentioned_word_4','foll_friends','N3_tweet_mentioned_word_5',
 'N2_tweet_mentioned_word_4','tweet_number_of_urls_std','N3_tweet_hastag_word_3','N1_retweet_hastag_word_1','hour_rt_tw_22',
 'daily_retweet_avg','hour_rt_tw_11','N1_tweet_word_5','retweet_number_of_mentions_std','N1_tweet_mentioned_word_7','N2_retweet_word_3',
 'N1_retweet_hastag_word_9','daily_rt_6','hour_rt_tw_6','hour_rt_20','retweet_number_of_mentions_avg','hour_rt_tw_20',
 'N2_tweet_mentioned_word_7','in_degree','N1_tweet_mentioned_tfidf','N2_retweet_hastag_tfidf','retweet_number_of_urls_std',
 'N3_retweet_mentioned_word_4','hour_rt_6','N2_tweet_mentioned_tfidf','hour_rt_1','daily_rt_tw_0','hour_rt_tw_1',
 'N1_tweet_hastag_word_1','N2_tweet_hastag_word_4','N2_retweet_mentioned_word_6','hour_rt_tw_23','N1_retweet_hastag_word_4',
 'N3_tweet_mentioned_word_7','N1_retweet_word_9','hour_rt_21','hour_rt_tw_16','hour_rt_13','hour_rt_16','daily_rt_2',
 'N1_tweet_hastag_word_9','N1_retweet_word_8','hour_rt_tw_12','daily_rt_5','N1_retweet_word_4','hour_rt_12',
 'N1_retweet_mentioned_word_0','N2_retweet_hastag_word_4','N2_retweet_hastag_word_1','N1_tweet_mentioned_word_5','daily_rt_tw_2',
 'retweet_number_of_urls_avg','N2_tweet_mentioned_word_5','daily_tw_2','hour_rt_tw_14','N3_tweet_mentioned_word_1',
 'N2_retweet_mentioned_word_2','N2_retweet_mentioned_word_9','N2_retweet_hastag_word_9','N2_tweet_mentioned_word_1','hour_rt_11',
 'N3_tweet_mentioned_word_3','N2_retweet_word_7','N1_tweet_hastag_word_0','N1_retweet_mentioned_word_8','daily_tweet_avg',
 'N3_retweet_hastag_word_8','N3_tweet_mentioned_word_0','daily_rt_3','retweet_time_min','N1_tweet_hastag_word_4',
 'N3_retweet_mentioned_word_6','N3_tweet_hastag_word_2','hour_rt_23','N1_retweet_word_2','hour_rt_tw_9','hour_rt_10',
 'tweet_number_of_hashtags_std','N2_tweet_hastag_word_8','daily_tw_0','N2_tweet_hastag_word_9','hour_rt_tw_7','verified',
 'N3_retweet_mentioned_word_9','hour_tw_23','hour_rt_tw_15','hour_rt_tw_21','N2_tweet_hastag_word_1','N3_tweet_word_3',
 'hour_rt_0','N3_tweet_word_7','hour_rt_15','hour_rt_tw_10','N1_tweet_mentioned_word_8','N3_retweet_word_0',
 'N1_retweet_word_3','N3_tweet_mentioned_word_9','daily_rt_4','hour_rt_tw_8','N2_retweet_mentioned_word_8',
 'N3_tweet_mentioned_word_2','hour_rt_tw_19','hour_rt_tw_4','N3_tweet_mentioned_word_6','N2_retweet_word_8',
 'N2_retweet_mentioned_tfidf','hour_rt_8','hour_tw_12','w_in_degree','N3_retweet_hastag_word_4',
 'N3_tweet_hastag_word_4','N2_retweet_word_1','N2_retweet_word_6','favourites_by_age','N1_retweet_word_0','N2_tweet_word_3',
 'hour_tw_2','N3_retweet_word_4','hour_rt_5','w_out_degree','N1_tweet_mentioned_word_3','N3_retweet_word_1',
 'daily_rt_tw_5','N3_retweet_hastag_word_1','N3_retweet_hastag_word_2','N3_tweet_hastag_word_7','daily_tw_6',
 'N2_tweet_hastag_word_2','N3_retweet_word_9','N3_retweet_word_8','retweet_number_of_hashtags_avg',
 'N2_tweet_mentioned_word_3','N3_tweet_word_5','hour_rt_7','hour_tw_14','N2_tweet_mentioned_word_2','hour_tw_0',
 'hour_rt_19','N3_tweet_hastag_word_8','N1_tweet_word_4','hour_rt_14','N3_tweet_word_1','N3_tweet_word_4',
 'name_and_screen_name_similarity','hour_rt_tw_5','tweet_retweet','hour_rt_3','N3_tweet_word_9',
 'N3_tweet_hastag_word_9','N2_tweet_word_0','N1_retweet_hastag_word_2','N2_retweet_word_5','N3_tweet_hastag_tfidf',
 'N2_retweet_hastag_word_0','tweet_number_of_hashtags_avg','N1_tweet_mentioned_word_2','w_degree','daily_tw_5',
 'N3_retweet_word_2','N2_tweet_word_5','N2_retweet_word_0','N3_tweet_word_6','N1_retweet_mentioned_word_2',
 'N2_tweet_mentioned_word_8','hour_tw_13','hour_rt_tw_0','hour_rt_18','N3_tweet_hastag_word_6','hour_rt_tw_18',
 'N1_tweet_word_6','N3_retweet_word_3','hour_tw_3','N1_retweet_word_5','hour_tw_5','N2_retweet_word_9',
 'daily_rt_1','hour_rt_2','N3_tweet_mentioned_word_8','N3_tweet_hastag_word_1','hour_rt_tw_17',
 'retweet_number_of_hashtags_std','N3_tweet_hastag_word_0','hour_rt_17','hour_rt_4','hour_rt_tw_2','N2_tweet_word_2',
 'daily_tw_4','N2_tweet_mentioned_word_0','N3_retweet_mentioned_word_2','N1_retweet_mentioned_tfidf',
 'hour_tw_17','N2_retweet_hastag_word_2','N2_tweet_word_4','N2_retweet_word_2','hour_tw_20',
 'N3_retweet_mentioned_word_8','N3_retweet_hastag_word_9','N2_tweet_mentioned_word_6','hour_tw_9','hour_tw_16','hour_tw_15',
 'N1_tweet_word_7','N3_retweet_word_6','N1_tweet_hastag_word_2','daily_tw_3','N1_retweet_word_6',
 'hour_tw_7','N1_tweet_mentioned_word_9','N2_tweet_word_9','N1_tweet_word_0','N1_retweet_word_7',
 'N2_tweet_hastag_word_0','N2_tweet_word_7','N3_retweet_mentioned_word_0','N1_tweet_word_9','N3_tweet_word_8',
 'hour_tw_11','N2_tweet_mentioned_word_9','N3_retweet_hastag_word_0','N3_retweet_word_5','statuses_count',
 'N1_retweet_word_1','hour_rt_tw_3','hour_tw_1','daily_tw_1','N1_retweet_mentioned_word_9','hour_tw_22',
 'hour_tw_19','N2_tweet_word_6','N1_tweet_mentioned_word_6','N1_tweet_word_8','hour_tw_18','hour_tw_21',
 'N2_tweet_word_8','hour_tw_10','N3_retweet_word_7','name_digits','N2_tweet_word_1','N1_tweet_word_1']
"""
nmbr_of_features = [80, 100, 119, 130, 149, 241, 304, 322]
nmbr_of_features = list(range(50, 150))
"""
#nmbr_of_features = list(range(158, 159))
#nmbr_of_features = list(range(119, 120))
#best nmbr of features 130 ....
#features = list(intersection_features)
features = freq_features_list
# rename target according to X values names (Shape remain the same)
"""
#filename = "../data/features_large_with_words.csv"
#filename = "../data/september_final_only_labeled_dataset.csv"
filename = "../data/september_old_labels.csv"
filename = "september_old_labels.csv"

def read_data(filename):
    suspend = [int(x.replace(" ","").replace("\t","")) for x in open("bots_suspend_remv","r").read().split("\n") if x != ""] 
    # read the csv file that was combined with word2vec and graph features
    df = pd.read_csv(filename, header=0)
    
    target = df["target"]
    keep_users = suspend + df[df["target"] == 0]["user_id"].to_list()
    df = df.loc[df["user_id"].isin(keep_users)]
    
    # random shuffle the dataframe
    df = shuffle(df)
    
    # extract user id from dataframe
    user_ids = df["user_id"]

    # extract target from datafrmae
    target = df["target"]
    
   

    df = df.drop(["user_id", "target"], axis="columns")

    # make stratified train and test split 80/20
    X_train, X_test, y_train, y_test = train_test_split(df,
                                                        target,
                                                        test_size=0.2,
                                                        stratify=target)

    return X_train, y_train, X_test, y_test



# scale  data of each portion of dataset (train, evaluation and test)
def scale_data(train, evaluation, test):
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
    # scale train and test
    X_eval = pd.DataFrame(scaler.transform(evaluation), columns=evaluation.columns)
    X_test = pd.DataFrame(scaler.transform(test), columns=test.columns)
    return X_train, X_eval, X_test


# oversample dataset portion
# used in separated form for train and evalution datasets
# do not use in test data at all
def oversample(data, target):
    smk = SMOTETomek()
    data, target = smk.fit_sample(data, target)
    return data, target


def get_most_freq_features(number_of_iter=20):
    model_scores = {"val": [], "train": [], "test": []}
    model_auc = {"val": [], "train": [], "test": []}
    feature_scores = []

    all_comp = number_of_iter * 5
    start_time = datetime.now()
    c = 0

    # Make multiple folds of data
    for fold in range(number_of_iter):
        # split entire dataset into 2 parts:
        #                    1st(80%) with all data used in training/validation
        #                    2nd(20%) with data used in final testing (20%)
        # Each time when we read file, we random shuffle it

        # df , target is 80% of dataset from file
        # df_test, target_test is 20% of dataset used only for evaluation
        df, target, df_test, target_test = read_data(filename, verbose=True)

        ##############################
        # Make 5 Fold cross validation#
        ##############################

        # Make Stratified K-Fold cross validation of K=5 with random shufle
        kf = StratifiedKFold(n_splits=5, shuffle=True)

        for train_index, test_index in kf.split(df, target):
            # i = 0
            X_train, X_val = df.iloc[train_index], df.iloc[test_index]
            y_train, y_val = target.iloc[train_index], target.iloc[test_index]

            ##############################################
            # Scale data (train,evaluation and final test)#
            ##############################################

            # scale train,evaluation and final testing data
            # scaler = DataScaler()
            # X_train = scaler.fit_transform(X_train)
            # X_val = scaler.transform(X_val)

            ####################################################
            # Oversample train and validation portion separately#
            ####################################################

            # oversampling clean on training set
            X_train, y_train = oversample(X_train, y_train)

            # oversampling clean on validation set
            X_val, y_val = oversample(X_val, y_val)

            ####################################################
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
            # Important, we do not oversample the final test set#
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
            ####################################################

            ############################################
            # Train model based on train portion of data#
            # This model used for feature selection.    #
            ############################################

            model = XGBClassifier(objective=objective,
                                  num_class=2,
                                  learning_rate=learning_rate,
                                  n_estimators=xgb_n_estimators,
                                  max_depth=max_depth,
                                  colsample_bytree=colsample_bytree,
                                  eval_metric=eval_metric,  # "rmse",
                                  tree_method="gpu_hist", #SERVER ONLY
                                  predictor = 'gpu_predictor',#SERVER ONLY
                                  use_label_encoder=False)
            model.fit(X_train, y_train)

            #########################################################
            # Select from trained model the N most important features#
            #########################################################

            selection = SelectFromModel(model, threshold=-np.inf, max_features=200, prefit=True)

            ######################################################################################
            # Based on those selected features, transform the train,validation and final test sets#
            ######################################################################################

            X_train = X_train.iloc[:, selection.get_support(indices=True)]
            X_val = X_val.iloc[:, selection.get_support(indices=True)]

            ########################################################################
            # Train model based on train portion of data with selected features only#
            ########################################################################

            selected_model = XGBClassifier(objective=objective,
                                           num_class=2,
                                           learning_rate=learning_rate,
                                           n_estimators=xgb_n_estimators,
                                           max_depth=max_depth,
                                           colsample_bytree=colsample_bytree,
                                           eval_metric=eval_metric,  # "rmse",
                                           tree_method="gpu_hist", #SERVER ONLY
                                           predictor = 'gpu_predictor',#SERVER ONLY
                                           use_label_encoder=False)

            XGB_fitted_opt = selected_model.fit(X_train, y_train)

            XGB_val_probs = XGB_fitted_opt.predict_proba(X_val)

            # keep probabilities for the positive outcome only

            XGB_val_probs = XGB_val_probs[:, 1]

            # XGB_precision_train, XGB_recall_train, _ = precision_recall_curve(y_train, XGB_train_probs)
            XGB_precision_val, XGB_recall_val, _ = precision_recall_curve(y_val, XGB_val_probs)

            # AUC

            XGB_auc = auc(XGB_recall_val, XGB_precision_val)

            print("Validation AUC:{}".format(XGB_auc))
            #########################

            ################################################
            # Evaluation of model by prediction of val data#
            ################################################

            # predict train labels
            predictions_train = selected_model.predict(X_train)

            # predict validation labels
            predictions_val = selected_model.predict(X_val)

            #########################
            # Compute accuracy scores#
            #########################

            #accuracy_train = accuracy_score(y_train, predictions_train)
            accuracy_val = accuracy_score(y_val, predictions_val)
            XGB_f1 = f1_score(y_val, predictions_val)
            print("Validation f1:{}".format(XGB_f1))

            # keep probabilities for the positive outcome only
            # XGB_train_probs = XGB_train_probs[:, 1]
            # XGB_val_probs = XGB_val_probs[:, 1]

            # XGB_precision_train, XGB_recall_train, _ = precision_recall_curve(y_train, XGB_train_probs)
            # XGB_precision_val, XGB_recall_val, _ = precision_recall_curve(y_val, XGB_val_probs)

            # AUC
            # XGB_auc_train = auc(XGB_recall_train, XGB_precision_train)
            # XGB_auc = auc(XGB_recall_val, XGB_precision_val)

            # model_auc["val"].append(XGB_auc * 100.0)
            # model_auc["train"].append(XGB_auc_train * 100.0)

            ########################################
            # Score model scores for particular fold#
            ########################################

            # model_scores["val"].append(accuracy_val * 100.0)
            # model_scores["train"].append(accuracy_train * 100.0)

            # Keep feature names and scores for each execution
            feature_scores.append((X_train.columns.tolist(),
                                   (accuracy_val + XGB_auc + XGB_f1) / 3.0
                                   ))
            # ,
            # accuracy_train * 100.0, accuracy_test * 100.0))

            c += 1
            spend_time = (datetime.now() - start_time).seconds / 60.0
            print("Time spend:{} min. left comp: {} left time:{}".format(spend_time, all_comp - c,
                                                                         (spend_time / c) * (all_comp - c)))
    # after multiple monte_carlo executions we need to identify list of sorted best features
    # sort iterational features by their validation score, where first items have highest val score
    feature_scores.sort(key=lambda t: t[1], reverse=True)
    freq_features = []
    for features, val_score in feature_scores[:10]:
        freq_features += features
    freq_features = [x[0] for x in Counter(freq_features).most_common()]
    return freq_features

def sort_features_by_score(features):
    iteration = 20
    feature_score = defaultdict(lambda: 0.0)

    for k in range(0, iteration):
        # read from file
        # --> 80% for train + validation (df and target)
        # --> 20% for testing (df_test and y_test) Used for Final Testing
        df, target, df_test, y_test = read_data(filename, scale=False, verbose=True)

        kf = StratifiedKFold(n_splits=5, shuffle=True)
        fold_id = 0
        for train_index, val_index in kf.split(df, target):
            X_train, X_val = df.iloc[train_index], df.iloc[val_index]
            y_train, y_val = target.iloc[train_index], target.iloc[val_index]


            # Oversample the Train and Validation data portions
            X_train, y_train = oversample(X_train, y_train)
            X_val, y_val = oversample(X_val, y_val)
            XGB_Y_train = y_train
            XGB_Y_val = y_val

            for feature in features:

                # Create new model which would be used for feature selection
                XGB_model = XGBClassifier(objective=objective,
                                          num_class=2,
                                          learning_rate=learning_rate,
                                          n_estimators=xgb_n_estimators,
                                          max_depth=max_depth,
                                          colsample_bytree=colsample_bytree,
                                          eval_metric=eval_metric,  # "rmse",

                                          tree_method="gpu_hist", #SERVER ONLY
                                          predictor='gpu_predictor',#SERVER ONLY
                                          use_label_encoder=False)


                XGB_X_train = X_train[feature]
                XGB_X_val = X_val[feature]

                XGB_fitted_opt = XGB_model.fit(XGB_X_train, XGB_Y_train.ravel())

                XGB_val_probs = XGB_fitted_opt.predict_proba(XGB_X_val)

                # keep probabilities for the positive outcome only
                XGB_val_probs = XGB_val_probs[:, 1]

                XGB_precision, XGB_recall, _ = precision_recall_curve(XGB_Y_val, XGB_val_probs)

                # AUC
                XGB_auc = auc(XGB_recall, XGB_precision)

                # F1 Score
                XGB_f1 = f1_score(XGB_Y_val, XGB_model.predict(XGB_X_val))

                # summarize scores
                # print('XGBoost validation: f1=%.3f auc=%.3f' % (XGB_f1, XGB_auc))
                # print('XGBoost test: f1=%.3f auc=%.3f' % (XGB_f1_test, XGB_auc_test))

                feature_score[feature] += (XGB_f1 + XGB_auc) / 2


            print("Fold:{} of 5  Iteration:{} of {}".format(fold_id, k, iteration))
            fold_id += 1

    feature_score = [ (feature, feature_score[feature] / (iteration * 5))for feature in features]
    feature_score.sort(key=lambda t:t[1], reverse=True)
    return [x[0] for x in feature_score]


def monte_carlo(features, nmbr_of_features = list(range(10, 250))):

    nmbr = [0] * len(features)

    f1nes = [0] * len(features)
    auces = [0] * len(features)
    f1_and_auc = [0] * len(features)

    f1nes_val = [0] * len(features)
    auces_val = [0] * len(features)
    f1_and_auc_val = [0] * len(features)

    f1nes_train = [0] * len(features)
    auces_train = [0] * len(features)
    f1_and_auc_train = [0] * len(features)

    iteration = 20

    for k in range(0, iteration):
        # read from file
        # --> 80% for train + validation (df and target)
        # --> 20% for testing (df_test and y_test) Used for Final Testing
        df, target, df_test, y_test = read_data(filename, scale=False, verbose=True)
        print("DF shape:{} test shape:{}".format(df.shape, df_test.shape))
        # split 80% data into 2 types : Train (X_train) (70%), Validation(X_val)(30%)
        # X_train, X_val, y_train, y_val = train_test_split(df, target, test_size=0.30,stratify=target)

        # Make Stratified K-Fold cross validation of K=5 with random shufle
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        fold_id = 0
        for train_index, val_index in kf.split(df, target):
            X_train, X_val = df.iloc[train_index], df.iloc[val_index]
            y_train, y_val = target.iloc[train_index], target.iloc[val_index]

            # Scale each data category (Train, Validaion and Test)by same scaller
            #scaler = DataScaler()
            #X_train = scaler.fit_transform(X_train)
            #X_val = scaler.transform(X_val)
            #X_test = scaler.transform(df_test)
            X_test = df_test
            # Oversample the Train and Validation data portions
            X_train, y_train = oversample(X_train, y_train)
            X_val, y_val = oversample(X_val, y_val)
            XGB_Y_train = y_train
            XGB_Y_val = y_val
            XGB_Y_test = y_test
            #for i in range(1, len(freq_features_list)):
            #for i in range(1, len(features)):


            for i in nmbr_of_features:

                # Create new model which would be used for feature selection
                XGB_model = XGBClassifier(objective=objective,
                                          num_class=2,
                                          learning_rate=learning_rate,
                                          n_estimators=xgb_n_estimators,
                                          max_depth=max_depth,
                                          colsample_bytree=colsample_bytree,
                                          eval_metric=eval_metric,  # "rmse",

                                          tree_method="gpu_hist", #SERVER ONLY
                                          predictor='gpu_predictor',#SERVER ONLY
                                          use_label_encoder=False)

                nmbr[i] = i
                XGB_X_train = X_train[features[:i]]
                XGB_X_val = X_val[features[:i]]
                XGB_X_test = X_test[features[:i]]

                XGB_fitted_opt = XGB_model.fit(XGB_X_train, XGB_Y_train.ravel())

                XGB_train_probs = XGB_fitted_opt.predict_proba(XGB_X_train)
                XGB_val_probs = XGB_fitted_opt.predict_proba(XGB_X_val)
                XGB_test_probs = XGB_fitted_opt.predict_proba(XGB_X_test)

                # keep probabilities for the positive outcome only
                XGB_train_probs = XGB_train_probs[:, 1]
                XGB_val_probs = XGB_val_probs[:, 1]
                XGB_test_probs = XGB_test_probs[:, 1]

                XGB_precision_train, XGB_recall_train, _ = precision_recall_curve(XGB_Y_train, XGB_train_probs)
                XGB_precision, XGB_recall, _ = precision_recall_curve(XGB_Y_val, XGB_val_probs)
                XGB_precision_test, XGB_recall_test, _ = precision_recall_curve(XGB_Y_test, XGB_test_probs)

                # AUC
                XGB_auc_train = auc(XGB_recall_train, XGB_precision_train)
                XGB_auc = auc(XGB_recall, XGB_precision)
                XGB_auc_test = auc(XGB_recall_test, XGB_precision_test)

                # F1 Score
                XGB_f1_train = f1_score(XGB_Y_train, XGB_model.predict(XGB_X_train))
                XGB_f1 = f1_score(XGB_Y_val, XGB_model.predict(XGB_X_val))
                XGB_f1_test = f1_score(XGB_Y_test, XGB_model.predict(XGB_X_test))

                # summarize scores
                # print('XGBoost validation: f1=%.3f auc=%.3f' % (XGB_f1, XGB_auc))
                # print('XGBoost test: f1=%.3f auc=%.3f' % (XGB_f1_test, XGB_auc_test))
                f1nes[i] += XGB_f1_test
                auces[i] += XGB_auc_test
                f1_and_auc[i] += (XGB_f1_test + XGB_auc_test) / 2

                f1nes_val[i] += XGB_f1
                auces_val[i] += XGB_auc
                f1_and_auc_val[i] += (XGB_f1 + XGB_auc) / 2

                f1nes_train[i] += XGB_f1_train
                auces_train[i] += XGB_auc_train
                f1_and_auc_train[i] += (XGB_f1_train + XGB_auc_train) / 2
            print("Fold:{} of 5  Iteration:{} of {}".format(fold_id, k, iteration))
            fold_id += 1

    for i in range(1, len(f1nes)):
        f1nes[i] /= (iteration * 5)
        auces[i] /= (iteration * 5)
        f1_and_auc[i] /= (iteration * 5)

        f1nes_val[i] /= (iteration * 5)
        auces_val[i] /= (iteration * 5)
        f1_and_auc_val[i] /= (iteration * 5)

        f1nes_train[i] /= (iteration * 5)
        auces_train[i] /= (iteration * 5)
        f1_and_auc_train[i] /= (iteration * 5)

    #### BEST N FEATURES 94,98
    f_out = open("FS_result_june_range_on_last_old_data.txt", "w+")
    f_out.write("Test data:\n")
    f_out.write("Max f1:{} for index:{} and auc:{}\n".format(max(f1nes),
                                                     f1nes.index(max(f1nes)),
                                                     auces[f1nes.index(max(f1nes))]
                                                     ))
    f_out.write("Max AUC:{} for index:{} and f1:{}\n".format( max(auces),
                                                      auces.index(max(auces)),
                                                      f1nes[auces.index(max(auces))]
                                                      ))
    f_out.write("Max Both AUC:{} for index:{} and f1:{}\n".format( auces[f1_and_auc.index(max(f1_and_auc))],
                                                          f1_and_auc.index(max(f1_and_auc)),
                                                          f1nes[f1_and_auc.index(max(f1_and_auc))]
                                                           ))

    f_out.write("Validation data:\n")
    f_out.write("Max f1:{} for index:{} and auc:{}\n".format( max(f1nes_val),
                                                      f1nes_val.index(max(f1nes_val)),
                                                      auces_val[f1nes_val.index(max(f1nes_val))]
                                                      ))
    f_out.write("Max AUC:{} for index:{} and f1:{}\n".format( max(auces_val),
                                                      auces_val.index(max(auces_val)),
                                                      f1nes_val[auces_val.index(max(auces_val))]
                                                      ))
    f_out.write("Max Both AUC:{} for index:{} and f1:{}\n".format( auces_val[f1_and_auc_val.index(max(f1_and_auc_val))],
                                                           f1_and_auc_val.index(max(f1_and_auc_val)),
                                                           f1nes_val[f1_and_auc_val.index(max(f1_and_auc_val))]
                                                          ))

    f_out.write("Train data:\n")
    f_out.write("Max f1:{} for index:{} and auc:{}\n".format( max(f1nes_train),
                                                      f1nes_train.index(max(f1nes_train)),
                                                      auces_train[f1nes_train.index(max(f1nes_train))]
                                                      ))
    f_out.write("Max AUC:{} for index:{} and f1:{}\n".format( max(auces_train),
                                                      auces_train.index(max(auces_train)),
                                                      f1nes_train[auces_train.index(max(auces_train))]
                                                      ))
    f_out.write("Max Both AUC:{} for index:{} and f1:{}\n".format( auces_train[f1_and_auc_train.index(max(f1_and_auc_train))],
                                                           f1_and_auc_train.index(max(f1_and_auc_train)),
                                                           f1nes_train[f1_and_auc_train.index(max(f1_and_auc_train))]
                                                           ))
    f_out.write("-------------------------------------\n")
    for fNumber in nmbr_of_features:
        f_out.write("Number of features:{}\n".format(fNumber))
        f_out.write("Train F1:{} AUC:{}\n".format(f1nes_train[fNumber], auces_train[fNumber]))
        f_out.write("Test F1:{} AUC:{}\n".format(f1nes[fNumber], auces[fNumber]))
        f_out.write("Validation F1:{} AUC:{}\n".format(f1nes_val[fNumber], auces_val[fNumber]))
        f_out.write("-------------------------------------\n")
    f_out.close()

import os, ast

if not os.path.isfile("june2021_freq_features"):
    print("File of freq features not exists")
    print("Starting identification of most frequent features")
    feature_list = get_most_freq_features()
    f_out = open("june2021_freq_features", "w+")
    f_out.write("{}".format(feature_list))
    f_out.close()
else:
    print("File of freq features exists, loading feature list...")
    feature_list = ast.literal_eval(open("june2021_freq_features", "r").read())

if not os.path.isfile("june2021_freq_features_sorted"):
    print("No file with sorted features are found")
    feature_list = sort_features_by_score(feature_list)
    f_out = open("june2021_freq_features_sorted", "w+")
    f_out.write("{}".format(feature_list))
    f_out.close()
else:
    print("Sorted features are loaded from file")
    feature_list = ast.literal_eval(open("june2021_freq_features_sorted", "r").read())

print("Start of feature measurement")
monte_carlo(feature_list)


