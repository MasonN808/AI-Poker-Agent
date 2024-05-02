import random
import sys
sys.path.append("./")
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.card import Card
from pypokerengine.engine.deck import Deck
from pypokerengine.engine.message_builder import MessageBuilder
from pypokerengine.utils.card_utils import gen_deck, gen_cards
from randomplayer import RandomPlayer
from pypokerengine.engine.round_manager import RoundManager

#TODO: Think about ways to make this smarter like configuring the random distributions away from random uniform across all potential values (e.g., skewed gaussian)

class State:
    def __init__(self, hole_card_main: list[str], community_cards: list[str], game_state: dict) -> None:
        self.hole_card_main = hole_card_main
        self.community_card = community_cards
        # self.opp_previous_action = ... # TODO Implement
        self.state_info = self.get_state_info_str(hole_card_main, community_cards)
        self.game_state = game_state
    
    @classmethod
    def from_game_state(self, game_state: dict):
        self.hole_cards_main = [str(i) for i in game_state["table"].seats.players[0].hole_card]
        # opp_hole = [str(i) for i in game_state["table"].seats.players[1].hole_card]
        # print(f"==>> opp_hole: {opp_hole}")
        # print(f"==>> self.hole_cards_main: {self.hole_cards_main}")
        self.community_cards = [str(i) for i in game_state["table"].get_community_card()]
        self.state_info = self.get_state_info_str(self.hole_cards_main, self.community_cards)
        self.game_state = game_state
        return self

    @staticmethod
    def get_state_info_str(hole_cards, community_cards):
        # Pad Community cards with 0s
        community_cards = community_cards + ['00'] * (5 - len(community_cards))
        state_info = f"{''.join(community_cards)}|{''.join(hole_cards)}"
        # Sort the card since done when populating dictionary
        sorted_card_string = sort_cards(state_info)
        return sorted_card_string
    
    @classmethod
    def random_state(self):
        """
        Generates a random state of the game.
        Since we can't initialize a game state along with the player's hoe cards, we do a hacky version
        """
        num_player = 2
        max_round = 10000000000
        small_blind_amount = 10
        ante = 0 
        emulator = Emulator()
        emulator.set_game_rule(num_player, max_round, small_blind_amount, ante)

        # 2. Setup GameState object
        p1_uuid = "uuid-1"
        p1_model = RandomPlayer(p1_uuid)
        emulator.register_player(p1_uuid, p1_model)
        p2_uuid = "uuid-2"
        p2_model = RandomPlayer(p2_uuid)
        emulator.register_player(p2_uuid, p2_model)
        players_info = {
            "uuid-1": { "name": "POMCP", "stack": 1000 },
            "uuid-2": { "name": "RANDOM", "stack": 1000 },
        }

        # Initializes the initial game state without dealing
        initial_state = emulator.generate_initial_game_state(players_info)
        # Actually starts the round and is now a player's turn
        # Hoe cards have been distributed, but not community cards
        game_state, events = emulator.start_new_round(initial_state)

        hole_cards_main = [str(i) for i in game_state["table"].seats.players[0].hole_card]

        # Create instance first then use it to call get_state_info_str
        self.hole_cards = hole_cards_main
        # print(f"==>> self.hole_cards: {self.hole_cards}")

        self.community_cards = []
        self.state_info = self.get_state_info_str(self.hole_cards, [])
        self.game_state = game_state

        # Community cards is empty
        return self, emulator

    def matches(self, hole_card_main=None, community_cards=None):
        if hole_card_main is not None and self.hole_card_main != hole_card_main:
            return False
        if community_cards is not None and self.community_card != community_cards:
            return False
        return True

def get_current_player_id(round_state):
    return round_state['next_player']

def get_valid_actions(game_state: dict) -> list[dict]:
    # Extracted from run_until_round_finish()
    next_player_pos = game_state["next_player"]
    # print(f"==>> next_player_pos: {next_player_pos}")
    msg = MessageBuilder.build_ask_message(next_player_pos, game_state)["message"]
    extracted_valid_actions = extract_valid_actions(msg["valid_actions"])
    assert extract_valid_actions != None, "valid actions should not be none"
    return extracted_valid_actions
 
def extract_valid_actions(valid_actions: list[dict]) -> list[str]:
    """
    Pypoker engine has actions: {'action': 'call'}
    This function extracts the action strings
    """
    extracted_actions = [action_dict['action'] for action_dict in valid_actions]
    return extracted_actions

def from_state_action_to_state(emulator: Emulator, game_state: dict, action: str):
    # TODO: check if emulator or RoundManger is correct since they both have apply_action()
    # MCTS works with RoundManager
    new_game_s, messages = emulator.apply_action(game_state, action)
    return new_game_s, messages

def add_state_tree_to_external(nodes: dict, state_info: str, tree) -> dict:
    """
    Addes a state-tree pair to dictionary for optimal action decision-making 
    that will be saved to json file which will be referenced during test-time 
    """
    # TODO: worried that if tree.state.state_info already exists. Might have to merge the trees which will mess up the tree
    # This is where I think POMCP solves this issue
    # TODO: Should we save all the possible trees as a list?
    # Sort the cards
    sorted_card_str = sort_cards(state_info)
    nodes[sorted_card_str] = tree
    return nodes

def sort_cards(card_string):
    suits_order = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
    ranks_order = {str(n): n for n in range(2, 10)}
    ranks_order.update({'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14})
    
    def card_key(card):
        # Extract rank and suit, e.g., 'H5' -> ('H', 5)
        suit, rank = card[0], card[1:]
        return (suits_order[suit], ranks_order[rank])
    
    # Split the string by '|'
    community, holes = card_string.split('|')
    
    # Sort community and hole cards
    sorted_community = ''.join(sorted((community[i:i+2] for i in range(0, len(community), 2)), key=card_key))
    sorted_holes = ''.join(sorted((holes[i:i+2] for i in range(0, len(holes), 2)), key=card_key))
    
    # Combine them back into a string with '|'
    return f"{sorted_community}|{sorted_holes}"

if __name__ == "__main__":
    card_string = "H5H3CK|C3S3"
    sorted_card_string = sort_cards(card_string)
    print(sorted_card_string)  # Output: CKH3H5|C3S3