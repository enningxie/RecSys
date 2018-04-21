#
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k

# load the movieLens 100K dataset.
# only five star rating are treated as positive
data = fetch_movielens(min_rating=5.0)

# Instantiate and train the model
model = LightFM(loss='warp')
model.fit(data['train'], epochs=30, num_threads=2)

# Evalute the trained model
test_precision = precision_at_k(model, data['test'], k=5).mean()

print(test_precision)