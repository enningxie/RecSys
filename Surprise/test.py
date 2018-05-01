from surprise import Dataset
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import KFold
from sklearn.model_selection import train_test_split

data = Dataset.load_builtin('ml-1m')

algo = SVD()

trainset = data.build_full_trainset()
algo.fit(trainset)

testset = trainset.build_testset()
predictions = algo.test(testset)

# RMSE
accuracy.rmse(predictions, verbose=True)