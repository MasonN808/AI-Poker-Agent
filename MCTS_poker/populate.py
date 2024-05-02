import copy
import random
import math
import sys

sys.path.insert(0, './')
from pypokerengine.engine.table import Table
from pypokerengine.engine.seats import Seats
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
# from pypokerengine.api.emulator import apply_action
from utils import State, add_state_tree_to_external, from_state_action_to_state, get_valid_actions
from pypokerengine.engine.round_manager import RoundManager
# from pypokerengine.engine.hand_evaluator import eval_hand
from pypokerengine.utils.game_state_utils import restore_game_state
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.api.emulator import Emulator
from randomplayer import RandomPlayer
import random as rand
from tqdm import tqdm

nodes = {}
state_actions = {}
class SearchTree:
    def __init__(self, player = None, state=None, action=None, visit=0, value=0, parent=None):
        self.player: str = player    # "main" or "opp"
        self.action: str = action    # Action by the opponent that was taken in parent
        self.parent: SearchTree = parent 
        self.visit: int = visit      # Number of visits
        self.value: int = value      # Value of node
        self.children: dict[SearchTree] = {}

        self.state: State = state          # Observation
        self.valid_actions: list[str] = None

    def expand(self, valid_actions: list[str]):
        assert valid_actions != None, "valid actions should not be none"
        for action in valid_actions:
            if self.player == "main":
                player = "opp"
            else:
                player = "main"

            self.children[action] = SearchTree(player=player, action=action, parent=self)

    def ucb(self, child):
        if child.visit == 0:
            return float('inf')  # Return a very large number to ensure this node gets selected
        else:
            return math.sqrt(math.log(self.visit) / child.visit)

class MCTS():
    """
    MCTS for Poker in pypoker engine
    """
    def __init__(self,
                 discount=0.8,
                 depth=0,
                 epsilon=1e-7,
                 explore=7,
                 n_particles=128):

        self.discount = discount
        self.depth = depth
        self.epsilon = epsilon
        self.explore = explore
        self.n_particles = n_particles
        self.tree = SearchTree()
        self.emulator = None
        # self.timeout = 500
        # self.timeout = 100_000
        self.timeout = 40_000_000

    # Search module
    def search(self, state=None):
        # Repeat Simulations until timeout
        for _ in tqdm(range(self.timeout), desc='Progress'):
            if state == None:
                # Sample an initial state (observation) and get the initialized pypoker emulator
                state, self.emulator = State.random_state()  # s ~ I(s_0=s)
            self.simulate(state, self.tree, 0)
        # Make a new hacshmap with state strings as keys and values as optimal actions
        print(f"Number of nodes: {len(nodes.items())}")
        for _ , (key, value) in enumerate(tqdm(nodes.items(), desc='Processing nodes')):
            if value.player == "opp":
                continue
            # If node has no children, take a random action
            r = rand.random()
            if value.children == {}:
                if r <= 0.5:
                    action = value.valid_actions[1]
                elif r<= 0.9 and len(value.valid_actions) == 3:
                    action = value.valid_actions[2]
                else:
                    action = value.valid_actions[0]
            # If at least one child has been traversed, then it has value
            # We choose the child with maximal value
            elif(any(hasattr(tree, 'visit') and getattr(tree, 'visit') == 0  for tree in value.children)):
                action = max(value.children.values(), key=lambda child: child.value).action
            # If all children have non-zero values use maximal value
            # TODO: could make this better since redundant as above elif
            else:
                action = max(value.children.values(), key=lambda child: child.value).action
            state_actions[key] = action

            # TODO: Some state_info is "|" with no community cards or hole cards because the player folded in prior action
        return state_actions

    def simulate(self, state, tree):
        """
        Simulation performed using the UCT Algorithm
        """
        global nodes

        if tree.state == None:
            tree.state = state
        if tree.valid_actions == None:
            tree.valid_actions = get_valid_actions(state.game_state)

        # Add the state and tree object to dictionary
        if tree.player == "main":
            nodes = add_state_tree_to_external(nodes, tree.state.state_info, tree)

        # Keep going down tree until a node with no children is found
        while tree.children:
            # Replace current node with the child that maximized UCB1(s) value
            child = max(tree.children.values(), key=lambda child: child.value + self.explore * tree.ucb(child))
            # Since some children may not have been initialized with state or valid actions
            if child.state == None:
                next_game_state , _ = from_state_action_to_state(self.emulator, tree.state.game_state, child.action)
                child.state = State.from_game_state(next_game_state)
            if child.valid_actions == None:
                child.valid_actions = get_valid_actions(child.state.game_state)

            tree = child
            
            # Add the state and tree object to dictionary
            if tree.player == "main":
                nodes = add_state_tree_to_external(nodes, tree.state.state_info, tree)

        # Now tree is assumed to be a leaf node
        # Check if the node has been traversed
        if tree.visit == 0:
            reward = self.rollout(tree.state, self.emulator)
        else:
            # If node has been visited, expand the tree and perform rollout
            tree.expand(tree.valid_actions)

            # Rollout on first child, other children will eventually get rolled out via UCB1
            action, child_tree = next(iter(tree.children.items()))
            # print("-------------------------------")
            # print(f"==>> tree.player: {tree.player}")
            # print(f"==>> tree.action: {tree.action}")
            # print(f"==>> tree.visit: {tree.visit}")
            # print(f"==>> tree.value: {tree.value}")
            # print(f"==>> tree.children: {tree.children}")
            # print(f"==>> tree.state: {tree.state}")
            # print(f"==>> tree.valid_actions: {tree.valid_actions}")
            # print(f"==>> ROUND FINISHED: {is_round_finish(tree.state.game_state)}")
            # print("-------------------------------")

            # Need to reset the players stack to prevent game from ending
            # TODO: Idk if this is right
            tree.state.game_state["table"].seats.players[0].stack = 1000
            tree.state.game_state["table"].seats.players[1].stack = 1000
            # Extract resulting state for child node after performing action from parent node
            next_game_state , messages = from_state_action_to_state(self.emulator, tree.state.game_state, action)
            # Check if next_game_state is end of round
            if is_round_finish(next_game_state):
                return

            tree = child_tree
            tree.state = State.from_game_state(next_game_state)
            tree.valid_actions = get_valid_actions(next_game_state)

            # Add the state and tree object to dictionary
            if tree.player == "main":
                nodes = add_state_tree_to_external(nodes, tree.state.state_info, tree)

            reward = self.rollout(tree.state, self.emulator)

        # Do backpropogation up the tree
        self.backup(tree, reward)

        return
    
    @staticmethod
    def backup(tree: SearchTree, reward: float):
        """
        Backpropogation step
        Assumption: 0-sum 2-player game
        """
        while tree is not None:
            tree.visit += 1
            # Assign negative reward to Opponent
            # Alternate the reward for 0-sum 2-player poker
            if tree.player == "opp":
                tree.value += reward
            else:
                tree.value -= reward
            
            tree = tree.parent
        
    def rollout(self, state: State, emulator: Emulator):
        emulator = copy.copy(emulator)
        cur_stack = state.game_state["table"].seats.players[0].stack
        end_game_state, events = emulator.run_until_round_finish(state.game_state)
        
        reward = end_game_state["table"].seats.players[0].stack - cur_stack
        return reward

def is_round_finish(game_state):
    return game_state["street"] != Const.Street.FINISHED

if __name__ == '__main__':
    from pypokerengine.api.emulator import Emulator
    import json

    mcts = MCTS()
    nodes = mcts.search()
       
    with open('search_tree_10M_sorted.json', 'w') as f:
        json.dump(nodes, f, indent=4)