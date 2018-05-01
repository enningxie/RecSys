import pickle
import os

import pandas as pd

from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise import dump
from surprise.accuracy import rmse

# load the dataset
train_file = os.path.expanduser('~') + '/.surprise_data/ml-100k/ml-100k/u1.base'
test_file = os.path.expanduser('~') + '/.surprise_data/ml-100k/ml-100k/u1.test'

data = Dataset.load_from_folds([(train_file, test_file)], Reader('ml-100k'))

# algo = KNNBasic()
#
# for trainset, testset in data.folds():
#     algo.train(trainset)
#     predictions = algo.test(testset)
#     rmse(predictions)
#
#     dump.dump('./dump_file', predictions, algo)



