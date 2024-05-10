import json

from POMCP_poker.utils import State
from pypokerengine.engine.card import Card
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.players import BasePokerPlayer

class Group14Player(BasePokerPlayer):

    def __init__(self, search_tree="BEST-search_tree_200000_reinvigoration-10_explore-100.json"):
        super().__init__()

        def remove_suits(cards: list[str]) -> list[str]:
            new_cards = []
            for card in cards:
                new_cards.append(card[1]) # Remove the suite
            return new_cards

        def sort_cards(card_string):
            ranks_order = {str(n): n for n in range(2, 10)}
            ranks_order.update({'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, '0': 15})
            
            def card_key(card):
                # Extract rank and suit, e.g., 'H5' -> ('H', 5)
                rank = card[0]
                return (ranks_order[rank])
            
            # Split the string by '|'
            community, holes = card_string.split('|')
            
            # Sort community and hole cards
            sorted_community = ''.join(sorted((community[i:i+1] for i in range(0, len(community), 1)), key=card_key))
            sorted_holes = ''.join(sorted((holes[i:i+1] for i in range(0, len(holes), 1)), key=card_key))
            
            # Combine them back into a string with '|'
            return f"{sorted_community}|{sorted_holes}"

        class State:
            def __init__(self, hole_card_main: list[str]=None, community_cards: list[str]=None, state_info: str=None, game_state: dict=None) -> None:
                self.hole_card_main = hole_card_main
                self.community_cards = community_cards
                if self.hole_card_main == None or self.community_cards == None:
                    self.state_info = None
                else:
                    self.state_info = self.get_state_info_str(hole_card_main, community_cards)
                self.game_state = game_state
            
            @staticmethod
            def get_state_info_str(hole_cards, community_cards):
                # REMOVE Suits
                trimmed_hole_cards = remove_suits(hole_cards)
                trimmed_community_cards = remove_suits(community_cards)
                # Pad Community cards with 0s
                community_cards = trimmed_community_cards + ['0'] * (5 - len(trimmed_community_cards))
                state_info = f"{''.join(community_cards)}|{''.join(trimmed_hole_cards)}"
                # Sort the card since done when populating dictionary
                sorted_card_string = sort_cards(state_info)
                return sorted_card_string
                
        self.search_tree = search_tree

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

            reward = self.hand_evaluator.eval_hand(hole_cards, community_cards)
            # Raise if above threshold
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
    return Group14Player()
