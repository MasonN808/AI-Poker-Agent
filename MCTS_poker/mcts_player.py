import json

import numpy as np
from MCTS_poker.utils import State
from pypokerengine.engine.card import Card
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.players import BasePokerPlayer
from time import sleep
import pprint
import pickle

from MCTS_poker.populate import SearchTree
import random as rand
import tensorflow as tf



class MCTSPlayer(BasePokerPlayer):
    def __init__(self, search_tree=None):
        super().__init__()
        self.search_tree = search_tree
        def load_model(model_path):
            return tf.keras.models.load_model(model_path)

        def load_encoder(encoder_path):
            with open(encoder_path, 'rb') as f:
                return pickle.load(f)

        def load_nodes():
            with open(self.search_tree, 'r') as f:
                state_actions = json.load(f)
            return state_actions

        self.state_actions = load_nodes()
        self.hand_evaluator = HandEvaluator()
        self.in_table = 0
        self.not_in_table = 0

    def declare_action(self, valid_actions, hole_card, round_state):
        # Check if current observation is a valid state in search tree
        obs = State.get_state_info_str(hole_cards=hole_card, community_cards=round_state["community_card"])
        # Check if the observation is in the keys
        if obs in self.state_actions.keys():
            self.in_table += 1
            return self.state_actions[obs]
        # Do the Heuristic Policy
        else:
            self.not_in_table += 1

            community_cards = [Card.from_str(str_card) for str_card in round_state["community_card"]]
            hole_cards = [Card.from_str(str_card) for str_card in hole_card]

            heuristic = self.hand_evaluator.eval_hand(hole_cards, community_cards)
            reward = heuristic
            if reward > 52000:
                for i in valid_actions:
                    if i["action"] == "raise":
                        action = i["action"]
                        return action
                action = valid_actions[1]["action"]
                return action
            # Just fold
            else:
                for i in valid_actions:
                    if i["action"] == "fold":
                        action = i["action"]
                        return action
        
    
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
