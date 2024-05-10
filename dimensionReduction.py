from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import SVD, NMF
from surprise import accuracy

# Load the built-in Movielens-100k dataset
data = Dataset.load_builtin('ml-100k')

# Split the dataset into train and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Initialize and train SVD algorithm
svd = SVD()
svd.fit(trainset)

# Make predictions using SVD
svd_predictions = svd.test(testset)

# Evaluate SVD performance
svd_rmse = accuracy.rmse(svd_predictions)

# Initialize and train NMF algorithm
nmf = NMF()
nmf.fit(trainset)

# Make predictions using NMF
nmf_predictions = nmf.test(testset)

# Evaluate NMF performance
nmf_rmse = accuracy.rmse(nmf_predictions)

print("SVD RMSE:", svd_rmse)
print("NMF RMSE:", nmf_rmse)
