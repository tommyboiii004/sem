from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD, accuracy

# Load the built-in Movielens-100k dataset
data = Dataset.load_builtin('ml-100k')

# Split the dataset into train and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Initialize SVD algorithm
svd = SVD()

# Train the algorithm on the trainset
svd.fit(trainset)

# Make predictions on the testset
predictions = svd.test(testset)

# Evaluate the performance of the algorithm
rmse = accuracy.rmse(predictions)

print("RMSE:", rmse)
