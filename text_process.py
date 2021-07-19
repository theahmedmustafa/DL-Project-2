# Imports
import json
import nltk
import numpy as np
from json import dump
from nltk.corpus import stopwords
from nltk.util import pr

def create_embedding(raw_text, label = None):
    # Get Text
    text = ('').join(raw_text.split('\n')).lower()

    # Tokenize Text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_list = stopwords.words('english')
    go_tokens = [word for word in tokens if word not in stop_list]

    # POS tagging
    focus_tags = ['NN', 'VB', 'JJ']
    tags = nltk.pos_tag(go_tokens)

    # Create embedding
    data = {"TEXT" : text}
    for pair in tags:
        word, tag = pair
        # Add to embedding if desired tag
        if tag[:2] in focus_tags:
            data[word] = 1 if word not in data.keys() else data[word] + 1

    # Get right Path
    if type(label) == str:
        PATH = 'embeddings/user_labels/' + label + '.json'
    else:
        PATH = 'embeddings/user_test/' + str(label) + '.json'

    # Save embedding
    with open(PATH, 'w+') as outfile:
        json.dump(data, outfile, indent = 4)
    outfile.close()

# Cosine Similarity Function
def get_cosine_similarity(ref, test):
    # Get Reference words
    ref_words = list(ref.keys())
    # Get Test words
    test_words = list(test.keys())
    # Get All words
    all_words = set(ref_words + test_words)
    # Get vectors
    vec_a, vec_b = [], []
    for word in all_words:
        if word == 'TEXT': continue
        a = 0 if word not in ref.keys() else ref[word]
        b = 0 if word not in test.keys() else test[word]
        vec_a.append(a); vec_b.append(b)

    # Convert to numpy
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    dot = np.dot(vec_a, vec_b)
    mag_a = np.linalg.norm(vec_a)
    mag_b = np.linalg.norm(vec_b)

    return dot / (mag_a * mag_b)

# Run File
if __name__ == '__main__':
    # Open Reference
    f = open("embeddings/user_labels/adidas.json")
    ref = json.load(f)
    f.close()

    # Open Test
    f = open("embeddings/user_test/1.json")
    test = json.load(f)
    f.close()

    score = get_cosine_similarity(ref, test)
    print(score)