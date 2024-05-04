import json

import numpy as np
from MCTS_poker.utils import State
from pypokerengine.players import BasePokerPlayer
from time import sleep
import pprint
import pickle

from MCTS_poker.populate import SearchTree
import random as rand
import tensorflow as tf



class MCTSPlayer(BasePokerPlayer):
    def __init__(self):
        def load_model(model_path):
            return tf.keras.models.load_model(model_path)

        def load_encoder(encoder_path):
            with open(encoder_path, 'rb') as f:
                return pickle.load(f)

        def load_nodes():
            with open('search_tree_4500000_reinvigoration-1000.json', 'r') as f:
            # with open('/nas/ucb/mason/AI-Poker-Agent/search_tree_40M_sorted.json', 'r') as f:
                state_actions = json.load(f)
            return state_actions
    
        super().__init__()
        self.state_actions = load_nodes()
        self.in_table = 0
        self.not_in_table = 0
        # Load the trained model
        # self.model = load_model('poker_decision_model_embedding.h5')
        # self.card_encoder = load_encoder("./MCTS_poker/mlp/card_encoder.pkl")
        # self.action_encoder = load_encoder("./MCTS_poker/mlp/action_encoder.pkl")


    def declare_action(self, valid_actions, hole_card, round_state):
        # Check if current observation is a valid state in search tree
        state = State.get_state_info_str(hole_cards=hole_card, community_cards=round_state["community_card"])
        if state in self.state_actions.keys():
            self.in_table += 1
            print(f"==>> self.in_table: {self.in_table}")
            print(f"==>> self.not_in_table: {self.not_in_table}")
            return self.state_actions[state]
        # Do a random action
        else:
            self.not_in_table += 1
            print(f"==>> self.in_table: {self.in_table}")
            print(f"==>> self.not_in_table: {self.not_in_table}")
            r = rand.random()
            if r <= 0.5:
                call_action_info = valid_actions[1]
            elif r<= 0.9 and len(valid_actions ) == 3:
                call_action_info = valid_actions[2]
            else:
                call_action_info = valid_actions[0]
            action = call_action_info["action"]
            return action  # action returned here is sent to the poker engine

            new_cards_encoded = self.card_encoder.transform([state])
            predictions = self.model.predict(new_cards_encoded)
            predicted_action_indices = np.argmax(predictions, axis=1)
            predicted_actions = self.action_encoder.inverse_transform(predicted_action_indices)
            print(f"==>> self.in_table: {self.in_table}")
            print(f"==>> self.not_in_table: {self.not_in_table}")

            print(predicted_actions)
            if predicted_actions not in valid_actions:
                r = rand.random()
                if r <= 0.5:
                    call_action_info = valid_actions[1]
                elif r<= 0.9 and len(valid_actions ) == 3:
                    call_action_info = valid_actions[2]
                else:
                    call_action_info = valid_actions[0]
                action = call_action_info["action"]

                return action  # action returned here is sent to the poker engine

            return predicted_actions
        
    
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return MCTSPlayer()
