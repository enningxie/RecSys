# describes how to test the performances of an algorithm on the trainset
from surprise import Dataset
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import KFold

data = Dataset.load_builtin('ml-1m')

algo = SVD()

trainset = data.build_full_trainset()
algo.fit(trainset)

testset = trainset.build_testset()
predictions = algo.test(testset)

# RMSE
accuracy.rmse(predictions, verbose=True)

# cross-validation
print('CV procedure:')

kf = KFold(n_splits=3)
for i, (trainset_cv, testset_cv) in enumerate(kf.split(data)):
    print('fold number', i + 1)
    algo.fit(trainset_cv)

    print('On testset,', end='  ')
    predictions = algo.test(testset_cv)
    accuracy.rmse(predictions, verbose=True)

    print('On trainset,', end='  ')
    predictions_ = algo.test(trainset_cv.build_testset())
    accuracy.rmse(predictions_, verbose=True)