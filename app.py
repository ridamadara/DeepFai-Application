import io
import librosa
import numpy as np
import math
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = load_model('model.h5')
fake = 0
real = 1

num_samples_per_segment = int(1.8 * 16000)  # 1.8 seconds of audio at 16kHz sampling rate
hop_length = 512

def calculate_fake_percentage(array):
  """Calculates the fake percentage in the array.

  Args:
    array: The array to calculate the fake percentage in.

  Returns:
    The fake percentage in the array.
  """

  number_of_real_values = 0
  number_of_fake_values = 0
  for value in array:
    if value == 0:
      number_of_fake_values += 1
    else:
      number_of_real_values += 1

  fake_percentage = number_of_fake_values / (number_of_real_values + number_of_fake_values) * 100
  return fake_percentage


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    audio_data, sr = librosa.load(file, sr=16000)

    # Split audio into segments
    num_segments = math.ceil(len(audio_data) / num_samples_per_segment)
    print(num_segments)
    segments = []
    for s in range(num_segments):
        start_sample = s * num_samples_per_segment
        finish_sample = min(start_sample + num_samples_per_segment, len(audio_data))
        segment = audio_data[start_sample:finish_sample]
        if segment.shape[0] == 28800:
            segments.append(segment)
            # print(segment.shape)
    # print(segments)

    # Perform deepfake detection on each segment
    predictions = []
    for segment in segments:
        mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, hop_length=hop_length)
        log_spectrogram = librosa.amplitude_to_db(mel_spectrogram)
        # print(log_spectrogram.shape)

        spectrogram_array = np.array(log_spectrogram)

        # Resize spectrogram to (128, 57, 1) shape
        resized_log_spectrogram = np.expand_dims(spectrogram_array, axis=-1)

        # shape: (128, 57, 3)
        resized_log_spectrogram = np.repeat(resized_log_spectrogram, 3, axis=-1)

        # print(resized_log_spectrogram.shape)

        # the prediction function expects 4d array as it can predict multiple samples
        resized_log_spectrogram = resized_log_spectrogram[np.newaxis, ...]  

        # Make prediction
        prediction = model.predict(resized_log_spectrogram)
        predicted_index = np.argmax(prediction, axis=1) 
        # prediction = model.predict(np.expand_dims(resized_log_spectrogram, axis=0))
        predictions.append(predicted_index[0])
    print(predictions)

    # Calculate overall detection percentage
    fake_percentage = calculate_fake_percentage(predictions)
    print(fake_percentage)
    
    return jsonify({'result': round(fake_percentage, 3)})   

if __name__ == '__main__':
    app.run()
