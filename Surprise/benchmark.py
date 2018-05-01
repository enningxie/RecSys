import time
import datetime
import random
import numpy as np
import six
from tabulate import tabulate

from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import CoClustering
from surprise import SlopeOne

# the algo to cross-validate
classes = (SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNWithMeans, KNNBaseline,
           CoClustering, BaselineOnly, NormalPredictor)

np.random.seed(0)
random.seed(0)

dataset = 'ml-1m'
data = Dataset.load_builtin(dataset)
kf = KFold(random_state=0)

table = []
for klass in classes:
    start = time.time()
    out = cross_validate(klass(), data, ['rmse', 'mae'], kf)
    cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))
    mean_rmse = '{:.3f}'.format(np.mean(out['test_rmse']))
    mean_mae = '{:.3f}'.format(np.mean(out['test_mae']))
    new_line = [mean_rmse, mean_mae, cv_time]
    print(tabulate([new_line], tablefmt="pipe"))
    table.append(new_line)

header = [
    'RMSE',
    'MAE',
    'Time'
]

print(tabulate(table, header, tablefmt='pipe'))

