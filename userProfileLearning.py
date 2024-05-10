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

# Get the user-item matrix
user_item_matrix = trainset.build_testset()

# Choose a user for profile learning (let's say user_id = 1)
user_id = str(1)

# Get the user profile (latent factors)
user_profile = svd.qi[trainset.to_inner_iid(user_id)]

print("User Profile for user", user_id, ":", user_profile)

