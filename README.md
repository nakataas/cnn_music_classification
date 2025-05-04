# CNN Music Genre Classification (GTZAN Dataset)

This project implements a Convolutional Neural Network (CNN) using Keras/TensorFlow to classify music genres based on audio features. It utilizes the GTZAN dataset and processes audio files into Mel Spectrograms for input into the CNN model. **This project was developed as an assignment for the MA4072 Deep Learning course.**

## Dataset

This project utilizes the **GTZAN Dataset - Music Genre Classification**. It contains 1000 audio tracks, each 30 seconds long, distributed across 10 distinct music genres.

* **Source:** Kaggle
* **Link:** [https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
* **Genres:** Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, Rock (implicitly, based on GTZAN standard)

The script uses the `kagglehub` library to download the dataset automatically.

## Process Overview

The project follows these main steps, as implemented in the script:

1.  **Data Loading:** Downloads the GTZAN dataset via `kagglehub`.
2.  **Preprocessing:**
    * Loads `.wav` audio files using `librosa`.
    * Converts audio signals into Mel Spectrograms (`n_fft=2048`, `hop_length=512`, `n_mels=128`).
    * Divides each 30-second track into 5 segments for data augmentation.
    * Saves the extracted Mel Spectrograms and corresponding labels into batched JSON files (`data_10_part_*.json`) to handle memory constraints during processing.
3.  **Model Building:**
    * Defines a Sequential CNN model using `tensorflow.keras`.
    * The architecture includes multiple blocks of Conv2D, MaxPooling2D, and BatchNormalization layers, followed by Flatten, Dense (with Dropout), and a final Softmax output layer for 10 classes.
4.  **Training:**
    * Loads data from the generated JSON files.
    * Splits data into training, validation, and test sets (80:20 split for train/test, then 20% of train for validation).
    * Compiles the model using the Adam optimizer (learning rate 0.0001) and `sparse_categorical_crossentropy` loss.
    * Trains the model for 30 epochs with a batch size of 32.
5.  **Evaluation:**
    * Evaluates the trained model on the unseen test set to calculate accuracy and loss.
    * Includes functionality to predict and compare actual vs. predicted genres for sample test data.
    * Saves the trained model to `cnn_music_genre_classification.h5`.

## Requirements

* Python 3.x
* TensorFlow (including Keras)
* Librosa
* NumPy
* Matplotlib
* Scikit-learn
* Kagglehub
* gdown (optional, for downloading pre-trained model example in comments)
* IPython (for audio display in notebooks)

You can typically install these using pip:
```bash
pip install tensorflow librosa numpy matplotlib scikit-learn kagglehub gdown ipython
