# Imports
import json
import spacy
from glob import glob
from NER.ner_functions import predict_brand
from text_process import get_cosine_similarity

# Predict File Function
def predict(FILE_PATH, labels, ner_model):
    # Load Json
    f = open(FILE_PATH)
    test = json.load(f)
    f.close()

    # TRY 1: NER PREDICTION
    text = test['TEXT']
    results = predict_brand(text, ner_model)
    # If results exist
    if results != None:
        for res in results:
            prediction, start, end, label = res
            # If Prediction exists in labels
            for label in labels:
                if label.find(prediction) >= 0: return prediction

    # TRY 2: COSINE SIMILARITY
    LABEL_FILES = glob('embeddings/user_labels/*')
    # Track Highest Similarity
    max_score, winner_index = 0, None
    # Find Similarities
    for i, FILE in enumerate(LABEL_FILES):
        # Get Reference embedding
        f = open(FILE)
        ref = json.load(f)
        f.close()
        # Get Cosine Similarity
        score = get_cosine_similarity(ref, test)
        # Update Maximum
        if score >= max_score:
            max_score = score
            winner_index = i
    # Return winner
    if max_score > 0.01: return labels[winner_index]
    else: return None

# Predict Test Files Functions
def get_predictions():
    # Get Brand Labels
    LABEL_FILES = glob('embeddings/user_labels/*')
    labels = []
    for FILE in LABEL_FILES:
        # Get File name
        filename = FILE.split('\\')[-1].split('.')[0]
        # Get brand name
        if filename.find('_') >= 0:
            parts = filename.split('_')
            brand = (' ').join(parts)
        else:
            brand = filename
        # Add to brands
        labels.append(brand)

    # NER Model
    model = spacy.load('NER/ner_model_brands/')

    # Test Transcripts
    TEST_FILES = glob('embeddings/user_test/*')

    # Perform Prediction
    predictions = []
    for FILE in TEST_FILES:
        prediction = predict(FILE, labels, model)
        if prediction == None: prediction = 'No ad detected / Inconclusive'
        predictions.append(prediction)

    return predictions

# Run File
if __name__ == '__main__':
    predictions = get_predictions()
    print(predictions)