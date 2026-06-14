import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load ratings

ratings = pd.read_csv("data/ratings.csv")

print(ratings.head())

# Surprise dataset

reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(
    ratings[['userId', 'movieId', 'rating']],
    reader
)

# Train/Test Split

trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

# Train SVD

model = SVD()

model.fit(trainset)

# Predict

predictions = model.test(testset)

# Evaluate

rmse = accuracy.rmse(
    predictions,
    verbose=True
)

print("RMSE:", rmse)
import pickle

pickle.dump(
    model,
    open("models/svd_model.pkl", "wb")
)

print("SVD Model Saved!")