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
    def __init__(self, search_tree="test.json"):
        super().__init__()
        self.search_tree = search_tree
        def load_model(model_path):
            return tf.keras.models.load_model(model_path)

        def load_encoder(encoder_path):
            with open(encoder_path, 'rb') as f:
                return pickle.load(f)

        def load_nodes():
            with open(self.search_tree, 'r') as f:
            # with open('search_tree_1000000_reinvigoration-1000__reinvigoration-5.json', 'r') as f:
                state_actions = json.load(f)
            return state_actions

        self.state_actions = load_nodes()
        self.hand_evaluator = HandEvaluator()
        self.in_table = 0
        self.not_in_table = 0
        # Load the trained model
        # self.model = load_model('poker_decision_model_embedding.h5')
        # self.card_encoder = load_encoder("./MCTS_poker/mlp/card_encoder.pkl")
        # self.action_encoder = load_encoder("./MCTS_poker/mlp/action_encoder.pkl")


    def declare_action(self, valid_actions, hole_card, round_state):
        # community_cards = [Card.from_str(str_card) for str_card in ["SA", "SK", "SQ", "SJ", "ST"]]
        # community_cards = [Card.from_str(str_card) for str_card in []]
        # hole_cards = [Card.from_str(str_card) for str_card in ["S9", "S8"]]
        # hole_cards = [Card.from_str(str_card) for str_card in ["S3", "S4"]]
        # heuristic = self.hand_evaluator.eval_hand(hole_cards, community_cards)
        # print("{0:b}".format(heuristic))
        # print(heuristic)
        # exit()
        # 100 0011 0100 0011
        # 1000 0000 1010 0000 1001 1000
        # Check if current observation is a valid state in search tree
        state = State.get_state_info_str(hole_cards=hole_card, community_cards=round_state["community_card"])
        # print(f"==>> self.in_table: {self.in_table}")
        # print(f"==>> self.not_in_table: {self.not_in_table}")
        if state in self.state_actions.keys():
            # print(f"==>> state IN keys: {state}")
            self.in_table += 1
            return self.state_actions[state]
        else:
        # Do a random action
            # print(f"==>> state NOT in keys: {state}")
            self.not_in_table += 1

            community_cards = [Card.from_str(str_card) for str_card in round_state["community_card"]]
            hole_cards = [Card.from_str(str_card) for str_card in hole_card]

            heuristic = self.hand_evaluator.eval_hand(hole_cards, community_cards)
            reward = heuristic
            if reward > 52000:
                for i in valid_actions:
                    if i["action"] == "raise":
                        action = i["action"]
                        return action  # action returned here is sent to the poker engine
                action = valid_actions[1]["action"]
                return action # action r
            # Just fold
            else:
                for i in valid_actions:
                    if i["action"] == "fold":
                        action = i["action"]
                        return action  # action returned here is sent to the poker engine
                
            # r = rand.random()
            # if r <= 0.5:
            #     call_action_info = valid_actions[1]
            # # The best defense against an always raising player is to always raise as well for the random policy
            # elif r<= 1 and len(valid_actions ) == 3:
            #     call_action_info = valid_actions[2]
            # else:
            #     call_action_info = valid_actions[0]
            # action = call_action_info["action"]
            # return action  # action returned here is sent to the poker engine
        
            # for i in valid_actions:
            #     if i["action"] == "raise":
            #         action = i["action"]
            #         return action  # action returned here is sent to the poker engine
            # action = valid_actions[1]["action"]
            # return action

            # new_cards_encoded = self.card_encoder.transform([state])
            # predictions = self.model.predict(new_cards_encoded)
            # predicted_action_indices = np.argmax(predictions, axis=1)
            # predicted_actions = self.action_encoder.inverse_transform(predicted_action_indices)
            # print(f"==>> self.in_table: {self.in_table}")
            # print(f"==>> self.not_in_table: {self.not_in_table}")

            # print(predicted_actions)
            # if predicted_actions not in valid_actions:
            #     r = rand.random()
            #     if r <= 0.5:
            #         call_action_info = valid_actions[1]
            #     elif r<= 0.9 and len(valid_actions ) == 3:
            #         call_action_info = valid_actions[2]
            #     else:
            #         call_action_info = valid_actions[0]
            #     action = call_action_info["action"]

            #     return action  # action returned here is sent to the poker engine

            # return predicted_actions
        
    
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
