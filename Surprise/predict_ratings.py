from surprise import KNNBasic
from surprise import Dataset

# load the dataset
data = Dataset.load_builtin('ml-1m')

# Retrieve the trainset
trainset = data.build_full_trainset()

# Build an algorithm and train it
algo = KNNBasic()
algo.fit(trainset)

uid = str(196)  # raw user id (as in the rating file). they are string
iid = str(302)

# get a prediction for specific users and items.
pred = algo.predict(uid, iid, r_ui=4, verbose=True)