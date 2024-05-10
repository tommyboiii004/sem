from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# Load the dataset
data = Dataset.load_builtin('ml-100k')

# Split the dataset into train and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Initialize and train the SVD algorithm
svd = SVD()
svd.fit(trainset)

# Evaluate the performance of the algorithm before the attack
rmse_before_attack = accuracy.rmse(svd.test(testset))

# Inject fake ratings for a specific item
testset += [('fake_user1', 'target_item', 5),
            ('fake_user2', 'target_item', 5),
            ('fake_user3', 'target_item', 1),
            ('fake_user4', 'target_item', 1)]

# Evaluate the performance of the algorithm after the attack
rmse_after_attack = accuracy.rmse(svd.test(testset))

print("RMSE before attack:", rmse_before_attack)
print("RMSE after attack:", rmse_after_attack)
