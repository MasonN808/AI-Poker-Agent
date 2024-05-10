import sys
sys.path.append("./")
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.card import Card
from pypokerengine.engine.message_builder import MessageBuilder
from pypokerengine.utils.game_state_utils import attach_hole_card, attach_hole_card_from_deck
from hand_eval_player import HeuristicPlayer

class State:
    def __init__(self, hole_card_main: list[str]=None, community_cards: list[str]=None, state_info: str=None, game_state: dict=None) -> None:
        self.hole_card_main = hole_card_main
        self.community_cards = community_cards
        if self.hole_card_main == None or self.community_cards == None:
            self.state_info = None
        else:
            self.state_info = self.get_state_info_str(hole_card_main, community_cards)
        self.game_state = game_state
    
    @classmethod
    def from_game_state(cls, game_state: dict):
        hole_cards_main = [str(i) for i in game_state["table"].seats.players[0].hole_card]
        community_cards = [str(i) for i in game_state["table"].get_community_card()]
        state_info = cls.get_state_info_str(hole_cards_main, community_cards)
        game_state_copy = game_state.copy()
        return cls(hole_card_main=hole_cards_main, community_cards=community_cards, state_info=state_info, game_state=game_state_copy)

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
    
    def random_state(self, state=None):
        """
        Generates a random state of the game.
        Since we can't initialize a game state along with the player's hoe cards, we do a hacky version
        """
        if state==None:
            num_player = 2
            max_round = 10000000000
            small_blind_amount = 10
            ante = 0 
            emulator = Emulator()
            emulator.set_game_rule(num_player, max_round, small_blind_amount, ante)

            # 2. Setup GameState object
            p1_uuid = "uuid-1"
            p1_model = HeuristicPlayer()
            emulator.register_player(p1_uuid, p1_model)
            p2_uuid = "uuid-2"
            p2_model = HeuristicPlayer()
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
            self.community_cards = []
            self.state_info = self.get_state_info_str(self.hole_cards, [])
            self.game_state = game_state

            # Community cards is empty
            return self, emulator
        
        else:
            num_player = 2
            max_round = 10000000000
            small_blind_amount = 10
            ante = 0 
            emulator = Emulator()
            emulator.set_game_rule(num_player, max_round, small_blind_amount, ante)

            # 2. Setup GameState object
            p1_uuid = "uuid-1"
            p1_model = HeuristicPlayer()
            emulator.register_player(p1_uuid, p1_model)
            p2_uuid = "uuid-2"
            p2_model = HeuristicPlayer()
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

            hole_cards_main = [str(i) for i in state.game_state["table"].seats.players[0].hole_card]

            # Set the players hole cards
            my_hole_card = state.game_state["table"].seats.players[0].hole_card
            my_uuid = state.game_state["table"].seats.players[0].uuid
            opp_uuid = state.game_state["table"].seats.players[1].uuid
            # game_state["table"].seats.players[0].hole_card = my_hole_card
            game_state = attach_hole_card(game_state, my_uuid, my_hole_card)
            game_state = attach_hole_card_from_deck(game_state, opp_uuid)

            # Create instance first then use it to call get_state_info_str
            self.hole_cards = hole_cards_main
            self.community_cards = []
            self.state_info = self.get_state_info_str(self.hole_cards, [])
            self.game_state = game_state

            # Community cards is empty
            return self, emulator

def get_valid_actions(game_state: dict) -> list[dict]:
    # Extracted from run_until_round_finish()
    next_player_pos = game_state["next_player"]
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
    # MCTS works with RoundManager
    new_game_s, messages = emulator.apply_action(game_state, action)
    return new_game_s, messages

def add_state_tree_to_external(nodes: dict, tree) -> dict:
    """
    Addes a state-tree pair to dictionary for optimal action decision-making 
    that will be saved to json file which will be referenced during test-time 
    """
    # Sort the cards just in case
    sorted_card_str = sort_cards(tree.state.state_info)
    # print(sorted_card_str)
    # Check if already in dict to append to list
    assert tree.player == "main", "Added trees should be from main player"
    if sorted_card_str in nodes:
        # This is to reduce memory consraints
        # Loop through all saved trees
        is_unique = True

        # Loop through all current belief states
        sorted_prunned_opp_hole_cards_tree = sort_cards_card_obj(tree.state.game_state["table"].seats.players[1].hole_card)
        for nodes_tree in nodes[sorted_card_str]:
            # Sort the opponnets hole cards and remove the suits
            sorted_prunned_opp_hole_cards_nodes = sort_cards_card_obj(nodes_tree.state.game_state["table"].seats.players[1].hole_card)
            # If the opponnet has the same hole cards as a tree in the nodes tree then break and dont add the tree to the trees list
            if (sorted_prunned_opp_hole_cards_tree == sorted_prunned_opp_hole_cards_nodes):
                is_unique = False
                break

        if is_unique:
            nodes[sorted_card_str].append(tree)
    else:
        nodes[sorted_card_str] = [tree]

    return nodes

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

def sort_cards_card_obj(cards: list[Card]):
    holes = [card.__str__() for card in cards]
    removed_suits = remove_suits(holes)
    ranks_order = {str(n): n for n in range(2, 10)}
    ranks_order.update({'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, '0': 15})
    
    def card_key(card):
        # Extract rank
        rank = card[0]
        return (ranks_order[rank])
    
    # Sort community and hole cards
    sorted_holes = sorted((removed_suits[i:i+1][0] for i in range(0, len(removed_suits))), key=card_key)
    
    # Combine them back into a string with '|'
    return sorted_holes

def remove_suits(cards: list[str]) -> list[str]:
    new_cards = []
    for card in cards:
        new_cards.append(card[1]) # Remove the suite
    return new_cards

if __name__ == "__main__":
    card_string = "53K00|A3"
    sorted_card_string = sort_cards(card_string)
    print(sorted_card_string)  # Output: CKH3H5|C3S3