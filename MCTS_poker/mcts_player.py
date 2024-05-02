import json
from MCTS_poker.utils import State
from pypokerengine.players import BasePokerPlayer
from time import sleep
import pprint
import pickle

from MCTS_poker.populate import SearchTree
import random as rand


class MCTSPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.state_actions = self.load_nodes()
        self.in_table = 0
        self.not_in_table = 0


    def load_nodes(self):
        with open('search_tree_40M_sorted.json', 'r') as f:
            state_actions = json.load(f)
        return state_actions

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
            r = rand.random()
            if r <= 0.5:
                call_action_info = valid_actions[1]
            elif r<= 0.9 and len(valid_actions ) == 3:
                call_action_info = valid_actions[2]
            else:
                call_action_info = valid_actions[0]
            action = call_action_info["action"]
            print(f"==>> self.in_table: {self.in_table}")
            print(f"==>> self.not_in_table: {self.not_in_table}")
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
    return MCTSPlayer()
