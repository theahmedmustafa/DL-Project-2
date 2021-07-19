# Imports
import os
import subprocess
from glob import glob
from re import VERBOSE
from pydub import AudioSegment
import speech_recognition as sr
 

# Audio Extraction Function
def get_wav(FILE_PATH):
    # MP3 Save Path
    MP3_PATH = "audio_cache/audio_mp3.mp3"
    # Remove existing MP3 file
    try:
        os.remove(MP3_PATH)
    except: pass
    # CMD FFMPEG MP3 Conversion
    command = "ffmpeg -i " + FILE_PATH + " " + MP3_PATH
    subprocess.check_output(command, shell = True, stderr = subprocess.DEVNULL)
    # Export Wav File
    mp3_file = AudioSegment.from_mp3(MP3_PATH)
    WAV_PATH = "audio_cache/audio_wav.wav"
    # Remove existing WAV file
    try:
        os.remove(WAV_PATH)
    except: pass
    # Save WAV file
    mp3_file.export(WAV_PATH, format = "wav")
    # Return Wav File
    return AudioSegment.from_wav(WAV_PATH)

# Split Function
def split(audio, split_time = 15):
    # Remove existing splits
    for file in glob('audio_cache/wav_splits/*'):
        os.remove(file)
    # Number of Splits
    total_duration =  len(audio)
    splits = int(total_duration / (split_time * 1000)) + 1
    # Split Audio
    for i in range(splits):
        # Time Points
        t1 = i * split_time * 1000
        t2 = (i + 1) * split_time * 1000
        if t2 > total_duration: t2 = total_duration
        # Split Audio
        split_audio = audio[t1:t2]
        # Ignore Too Small File (for last file)
        if len(split_audio) < 3000: continue
        # Save Audio
        EXPORT_PATH = 'audio_cache/wav_splits/split_' + str(i + 1) + '.wav'
        split_audio.export(EXPORT_PATH, format="wav")
    # Return File Paths
    return glob('audio_cache/wav_splits/*')

# Speech to Text Function
def speech_to_text(PATH):
    # Initialize Recognizer
    recognizer = sr.Recognizer()
    # Begin Transcription
    with sr.AudioFile(PATH) as source:
        # Listen Source File
        audio = recognizer.listen(source)
        try:
            # Google Transcription
            text = recognizer.recognize_google(audio)
        except Exception as e:
            print('Transcription Failed')
            text = ''
    # Return Text
    return text