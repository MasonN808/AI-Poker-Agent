import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def train():
    # Load data
    with open('search_tree_40M_sorted.json', 'r') as f:
        data = json.load(f)

    # Extract inputs and labels
    inputs = list(data.keys())
    labels = list(data.values())

    # Tokenization and integer encoding
    tokenizer = Tokenizer(char_level=True)  # Set to False if you want word-level tokenization
    tokenizer.fit_on_texts(inputs)
    sequences = tokenizer.texts_to_sequences(inputs)
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because indexing starts at 1

    # Padding sequences
    max_length = max(len(seq) for seq in sequences)
    X = pad_sequences(sequences, maxlen=max_length, padding='post')

    # Encode labels
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(np.array(labels).reshape(-1, 1))

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Neural Network Model
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=10, input_length=max_length),
        LSTM(50),
        Dense(Y.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the Model
    model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")


    # Save model
    model.save('poker_decision_model_lstm.h5')

if __name__ == "__main__":
    train()
