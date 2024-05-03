import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Embedding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import FeatureHasher
import json



def preprocess_data(data_path):
    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Prepare data
    cards = list(data.keys())
    actions = list(data.values())

    # Encode cards
    card_encoder = LabelEncoder()
    X = card_encoder.fit_transform(cards)
    # Saving the encoder
    with open('/nas/ucb/mason/AI-Poker-Agent/MCTS_poker/mlp/card_encoder.pkl', 'wb') as f:
        pickle.dump(card_encoder, f)

    # Encode actions
    action_encoder = LabelEncoder()
    y = action_encoder.fit_transform(actions)
    y = tf.keras.utils.to_categorical(y)  # One-hot encode the labels

    with open('/nas/ucb/mason/AI-Poker-Agent/MCTS_poker/mlp/action_encoder.pkl', 'wb') as f:
        pickle.dump(action_encoder, f)

    return X, y, len(card_encoder.classes_), len(action_encoder.classes_)

def create_model(num_cards, num_actions):
    model = Sequential([
        Embedding(input_dim=num_cards, output_dim=64),  # 50 can be adjusted based on model complexity and dataset
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_actions, activation='softmax')  # Change to match the number of unique actions
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train():
    X, y, num_cards, num_actions = preprocess_data('search_tree_40M_sorted.json')
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    # Reshape input for embedding
    X_train = X_train.reshape(-1, 1)
    X_val = X_val.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    # Create and train the model
    model = create_model(num_cards, num_actions)
    model.fit(X_train, y_train, epochs=10, batch_size=100, validation_data=(X_val, y_val))

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Accuracy:", accuracy)

    # Save model
    model.save('mlp/poker_decision_model_embedding.h5')

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    train()
