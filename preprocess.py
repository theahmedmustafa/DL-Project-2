# Imports
from glob import glob
from tqdm import tqdm
from audio_process import *
from text_process import *

# Process Single Video Function
def preprocess_video(VIDEO_PATH, isLabel = True, test_split_time = 20):
    # Audio Extraction from Video
    print('EXTRACTING AUDIO')
    audio = get_wav(VIDEO_PATH)

    # PROCESS AS USER LABEL
    if isLabel:
        # Split Audio
        FILES = split(audio)

        # Transcribe Audio
        print('CREATING TRANSCRIPT')
        text = ''
        for PATH in tqdm(FILES):
            text += speech_to_text(PATH) + '\n'

        # Get Label
        label = VIDEO_PATH.split('\\')[-1].split('.')[0]

        # Save Embedding
        create_embedding(text, label)
        print('EMBEDDING SAVED \n')

    # PROCESS AS USER TEST
    else:
        # Split Audio
        FILES = split(audio, test_split_time)

        # Transcribe Audio
        print('CREATING TRANSCRIPTS')
        for i, PATH in enumerate(tqdm(FILES)):
            text = speech_to_text(PATH)
            if len(text) > 0: create_embedding(text, i+1)
        print('EMBEDDINGS SAVED \n')

# Process User Inputs Function
def preprocess_user_labels():
    PATHS = glob('video_files/user_labels/*')
    for PATH in PATHS:
        # Preprocess
        preprocess_video(PATH)

# Process User Test Function
def preprocess_user_test(test_split_time = 20):
    PATHS = glob('video_files/user_test/*')
    for PATH in PATHS:
        preprocess_video(PATH, False, test_split_time)

# Run File
if __name__ == '__main__':
    preprocess_user_test()