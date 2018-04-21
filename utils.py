import argparse
import pandas as pd
import numpy as np


# parameters
def arg_parser():
    args = argparse.ArgumentParser()
    args.add_argument('--ratings_path', type=str, default='./ml-1m/ratings.dat')
    args.add_argument('--pd_path', type=str, default='./files/ratings.csv')

    return args.parse_args()


# read data from the file.
def load_data(path):
    with open(path, 'r') as f:
        data_lines = f.readlines()
    pd_data = _put_data_to_pd(data_lines)
    return pd_data


def _put_data_to_pd(data):
    user_id, movie_id, ratings, timestamp = [], [], [], []
    for i in data:
        tmp_list = i.split('::')
        user_id.append(tmp_list[0])
        movie_id.append(tmp_list[1])
        ratings.append(tmp_list[2])
        timestamp.append(tmp_list[3][:-1])
    dict_pd = dict()
    dict_pd['UserID'] = user_id
    dict_pd['MovieID'] = movie_id
    dict_pd['Rating'] = ratings
    dict_pd['Timestamp'] = timestamp
    pd_data = pd.DataFrame(dict_pd, dtype=np.int)
    return pd_data


def save_pd(pd_data, pd_path):
    pd_data.to_csv(pd_path, index=False)
    print('Saved finished!')


# return a dict,
# {user_id: [movie_id1, movie_id2, movie_id3, ...]}
def get_dict_from_pd(pd_path):
    ratings_dict = dict()
    ratings_pd = pd.read_csv(pd_path)
    ratings_pd_grouped = ratings_pd.groupby(['UserID'])
    for k in ratings_pd_grouped.groups.keys():
        ratings_dict[k] = list(ratings_pd_grouped.get_group(k).sort_values(by='Timestamp')['MovieID'])
    return ratings_dict


if __name__ == '__main__':
    FLAGS = arg_parser()
    ratings_dict_ = get_dict_from_pd(FLAGS.pd_path)
    print(len(ratings_dict_.get(14, 'no_idea.')))