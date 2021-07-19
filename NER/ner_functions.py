import random
import spacy
from spacy.training.example import Example
from NER.process_transcripts import get_train_dataset

# Train Model Function
def train_ner(epochs, dropout, save_file):
    # Get complete dataset from file
    FILE_PATH = 'transcripts.xlsx'
    DATASET = get_train_dataset(FILE_PATH)

    # Load SpaCy
    nlp = spacy.load('en_core_web_sm')

    # Add labels to "ner"
    ner = nlp.get_pipe("ner") # get the NER pipeline
    for _, annotations in DATASET:
      for ent in annotations.get("entities"):
        ner.add_label(ent[2])

    # Disable unneeded pipeline components
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

    # Training the model from TRAIN_DATA
    TRAIN_DATA = DATASET
    with nlp.disable_pipes(*unaffected_pipes):
        for epoch in range(epochs):
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = spacy.util.minibatch(TRAIN_DATA, size = 50)

            for batch in batches:
                for text, annotations in batch:
                    doc = nlp.make_doc(text) # create example
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], losses=losses, drop=dropout) # update the model
                    print( "Epoch:", epoch, "\tLoss:", round(losses['ner'], 2))

    # Save model to disk
    nlp.to_disk(save_file)

# Predict Brand Function
def predict_brand(statement, model):
    # Find Brand
    doc = model(statement)
    # If no brand found
    if  len(doc.ents) == 0:
        return None
    # Results
    res = []
    for ent in doc.ents:
        res.append((ent.text, ent.start_char, ent.end_char, ent.label_))
    return res

# Run File
if __name__ == '__main__':
    # Parameters
    epochs = 27
    dropout = 0.65
    model_name = 'NER/ner_model_brands/'

    # Train Model
    training = False
    if training:
        train_ner(epochs, dropout, model_name)

    # Load Model
    model = spacy.load(model_name)

    # Test NER
    statements = [
    "it's a good time for the great taste of mcdonald's quarter pounder",
    "the gillette fusion offers a totally new shaving experience",
    "with wix you can create your own professional website for your business",
    "thats why i take tylenol the rapid release gel relieves pain fast so i can sleep",
    "it’s confidence, it’s belief, it’s a way of life, it’s nike",
    "writing is not easy that's why grammarly can help",
    "revolutionalize reading with amazon kindle",
    "with the karlstad series you have the flexibility to create the perfect seating combination",
    "at apple we guarantee you expensive items with shit service",
    "mcdonald's is a company. I like samsung",
    "I have a bmw i8"
    ]
    
    for statement in statements:
        # Make Brand Prediction
        predictions = predict_brand(statement, model)
        # If no prediction
        if predictions == None:
            print('NO BRAND FOUND')
            continue
        # Print Predictions
        for prediction in predictions:
            text, start, end, label = prediction
            print(text, start, end, label)