import numpy as np
from pypokerengine.players import BasePokerPlayer
from pomcp import POMCP, Generator

class POMCPPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.ab = POMCP(Generator, gamma=0.9, timeout=1000, no_particles=300)
        self.action_history = []
        self.observation_history = []

    def declare_action(self, valid_actions, hole_card, round_state):
        # Here we simplify actions to a tuple (action, amount) or a string if no amount is necessary
        current_observation = self.generate_observation(hole_card, round_state)
        self.ab.initialize(state=self.generate_initial_state(round_state), action_space=[a['action'] for a in valid_actions], observation=current_observation)
        action_idx = self.ab.Search()
        action = valid_actions[action_idx]['action']
        amount = valid_actions[action_idx]['amount']
        return action, amount

    def generate_initial_state(self, round_state):
        # Customize this function to convert round_state to your POMCP state representation
        return None

    def generate_observation(self, hole_card, round_state):
        # Generate an observation from the current state of the game
        return hole_card + str(round_state)

    def receive_game_update_message(self, action, round_state):
        # This is called whenever any player takes an action
        observation = self.generate_observation(self.hole_card, round_state)
        self.ab.tree.prune_after_action(action, observation)
        self.ab.UpdateBelief(action, observation)
        self.action_history.append(action)
        self.observation_history.append(observation)

    def receive_round_result_message(self, winners, hand_info, round_state):
        # Use this to update the tree or reset the tree at the end of a round if necessary
        pass

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

if __name__ == "__main__":
    from pypokerengine.api.game import setup_config, start_poker
    # Setup config
    config = setup_config(max_round=10, initial_stack=10000, small_blind_amount=20)
    config.register_player(name="AI_1", algorithm=POMCPPlayer())
    config.register_player(name="AI_2", algorithm=POMCPPlayer())

    # Start the poker game
    game_result = start_poker(config, verbose=1)
