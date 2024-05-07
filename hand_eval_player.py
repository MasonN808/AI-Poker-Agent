import json

import numpy as np
from MCTS_poker.utils import State
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.players import BasePokerPlayer
from time import sleep
import pprint
import pickle

from MCTS_poker.populate import SearchTree
import random as rand
import tensorflow as tf
from pypokerengine.engine.card import Card

class HeuristicPlayer(BasePokerPlayer):
    def __init__(self, epsilon=52000):
        # epsilon = 52000 
        # 1100 1011 0010 0000
        # raise if hole card better than high card Q
        super().__init__()
        self.hand_evaluator = HandEvaluator()
        self.raises = 0
        self.folds = 0
        self.calls = 0
        self.highest_value = 0
        self.lowest_value = 0
        self.epsilon = epsilon


    def declare_action(self, valid_actions, hole_card, round_state):
        community_cards = [Card.from_str(str_card) for str_card in round_state["community_card"]]
        hole_cards = [Card.from_str(str_card) for str_card in hole_card]

        heuristic = self.hand_evaluator.eval_hand(hole_cards, community_cards)
        reward = heuristic

        # print(f"==>> self.raises: {self.raises}")
        # print(f"==>> self.folds: {self.folds}")
        # hand_info = HandEvaluator.gen_hand_rank_info(hole_cards, community_cards)
        # hand_strength = hand_info['hand']['strength']
        if reward > self.highest_value:
            self.highest_value = reward
        if reward < self.lowest_value:
            self.lowest_value = reward
        # print(f"==>> self.lowest_value: {self.lowest_value}")
        # print(f"==>> self.highest_value: {self.highest_value}")
        if reward > self.epsilon:
            for i in valid_actions:
                if i["action"] == "raise":
                    action = i["action"]
                    self.raises += 1
                    return action  # action returned here is sent to the poker engine
            action = valid_actions[1]["action"]
            return action # action r
        # Just fold
        else:
            for i in valid_actions:
                if i["action"] == "fold":
                    action = i["action"]
                    self.folds += 1
                    return action  # action returned here is sent to the poker engine
        
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
    return HeuristicPlayer()
