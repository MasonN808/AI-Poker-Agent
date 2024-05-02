import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import json

def train():
    # Load data
    with open('search_tree_40M_sorted.json', 'r') as f:
        data = json.load(f)

    # Prepare data
    cards = list(data.keys())
    actions = list(data.values())
    card_encoder = OneHotEncoder(sparse_output=True)
    X = card_encoder.fit_transform(np.array(cards).reshape(-1, 1))

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(actions)
    y = tf.keras.utils.to_categorical(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)  # further split for validation

    # Define model
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, epochs=10, batch_size=100, validation_data=(X_val, y_val))

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Accuracy:", accuracy)

    # Save model
    model.save('poker_decision_model_conv.h5')

if __name__ == "__main__":
    train()
