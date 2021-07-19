# Imports
from preprocess import *
from prediction import *

# Create User Labels Embeddings
print('\n------- PREPROCESSING USER LABELLED VIDEOS -------\n')
#preprocess_user_labels()

# Create User Test Embeddings
# -------- REMOVE TEST EMBEDDINGS BEFORE USING!! ------------
print('\n------- PREPROCESSING USER TEST VIDEO -------\n')
test_split_time = 20
#preprocess_user_test(test_split_time)

# Perform Predictions
predictions = get_predictions()
print('\n------- TEST VIDEO PREDICTION -------\n')
for i, prediction in enumerate(predictions):
    print(f'{i * test_split_time} to {(i + 1) * test_split_time} secs:  {prediction}')
print('\n')