from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import Dataset
from surprise.model_selection import cross_validate

# Load dataset
data = Dataset.load_builtin('ml-1m')

# Example using ALS
print('Using ALS')
bsl_options = {
    'method': 'als',
    'n_epochs': 5,
    'reg_u': 12,
    'reg_i': 5
}

algo = BaselineOnly(bsl_options=bsl_options)

cross_validate(algo, data, verbose=True)

# Example using SGD
print('Using SGD')
bsl_options = {
    'method': 'sgd',
    'learning_rate': 0.0005
}

algo = BaselineOnly(bsl_options=bsl_options)

cross_validate(algo, data, verbose=True)

# some similarity measures may use baselines. It works just the same.
print('Using ALS with pearson_baseline similarity')
bsl_options = {
    'method': 'als',
    'n_epochs': 20
}

sim_options = {'name': 'pearson_baseline'}
algo = KNNBasic(bsl_options=bsl_options, sim_options=sim_options)

cross_validate(algo, data, verbose=True)
