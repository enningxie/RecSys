from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

# Load data
data = Dataset.load_builtin('ml-1m')

# SVD algo
algo = SVD()

# run 5-fold cross-validation and print the result
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


