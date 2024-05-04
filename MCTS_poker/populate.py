from collections import Counter
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
from pypokerengine.engine.hand_evaluator import HandEvaluator
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
        self.emulator = None
        self.hand_evaluator = HandEvaluator()

        self.num_rollouts = 5
        # self.timeout = 5000
        # self.timeout = 200_000
        # self.timeout = 4_500_000
        self.timeout = 2_00_000
        # self.timeout = 50_000_000
        self.reinvigoration = 1000

    # Search module
    def search(self, state=None):
        global nodes
        # NOTE: Some state_info is "|" with no community cards or hole cards because the player folded in prior action
        # Repeat Simulations until timeout
        for t in tqdm(range(self.timeout), desc='Progress'):
            # print(t)
            # TODO: mason-decide if this is right or smaple from new state after each simulate()
            if state == None:
                # print("new state")
                # Sample an initial state (observation) and get the initialized pypoker emulator
                state_instance = State()
                state, self.emulator = state_instance.random_state()  # s ~ I(s_0=s)
            # print(f"==>> state: {state.state_info}")
            if state.game_state['next_player'] == 1:
                player = "main"
            else:
                player = "opp"
            
            # Check if state info is a key in nodes to avoid creating unnecessary new tree objects
            if state.state_info in nodes and player == "main":
                # Check if any trees in value-tree list has the same opponent hole card
                opp_hole_cards_and_trees = [(tree.state.game_state["table"].seats.players[1].hole_card, tree) for tree in nodes[state.state_info]]
                # Now get trees that have the same community card and opponnent hole card
                trees = [tup[1] for tup in opp_hole_cards_and_trees if 
                         (tup[1].state.community_cards == state.community_cards) and  # Check community cards are the same
                         (tup[0] == state.game_state["table"].seats.players[1].hole_card)] # Check hole card of opponenet is the same
                # TODO: Figure out this assertion
                # assert len(trees) <= 1, "Tree should only be one, otherwise, we have duplicate states in dictionary NOT GOOD!"
                if len(trees) >= 1:
                    while trees:
                        self.simulate(state, trees.pop())
                    # Sample from the start state every n simulations
                    if t % self.reinvigoration == 0:
                        state = None 
                    # Skip outside simulate
                    continue
                else:
                    # assert len(trees) != 0, "Length of trees should never be 0!"
                    tree = SearchTree(player=player, state=state, action=None, parent=None)
            else:
                tree = SearchTree(player=player, state=state, action=None, parent=None)

            self.simulate(state, tree)

            # Sample from the start state every n simulations
            if t % self.reinvigoration == 0:
                state = None 

        # Make a new hacshmap with state strings as keys and values as optimal actions
        print(f"Number of nodes: {len(nodes.items())}")
        many_trees = 0
        singleton_trees = 0
        for _ , (key, trees) in enumerate(tqdm(nodes.items(), desc='Processing nodes')):
            optimal_actions = []

            if len(trees) == 1:
                singleton_trees += 1
            else:
                many_trees += 1

            for tree in trees:
                assert tree.state.state_info == key

                # If node has no children, take a random action
                if tree.children == {}:
                    pass
                else:
                    action = max(tree.children.values(), key=lambda child: child.value).action
                    optimal_actions.append(action)

            # If none of the trees have children perform random action
            if optimal_actions == []:
                r = rand.random()
                if r <= 0.5:
                    action = tree.valid_actions[1]
                elif r<= 0.9 and len(tree.valid_actions) == 3:
                    action = tree.valid_actions[2]
                else:
                    action = tree.valid_actions[0]
                state_actions[key] = action
            # Select the action that appears the most amount of times
            else:
                # Count the occurrences of each element
                counter = Counter(optimal_actions)

                # Find the maximum count
                max_count = max(counter.values())

                most_common_actions = [element for element, count in counter.items() if count == max_count]

                random_most_common_action = random.choice(most_common_actions)

                state_actions[key] = random_most_common_action
        print(f"==>> many_trees: {many_trees}")
        print(f"==>> singleton_trees: {singleton_trees}")
                
        return state_actions

    def simulate(self, state, tree):
        """
        Simulation performed using the UCT Algorithm
        """
        global nodes

        assert tree.state != None, "State is None"
        if tree.valid_actions == None:
            tree.valid_actions = get_valid_actions(state.game_state)

        # Keep going down tree until a node with no children is found
        while tree.children:
            # Replace current node with the child that maximized UCB1(s) value
            child = max(tree.children.values(), key=lambda child: child.value + self.explore * tree.ucb(child))
            # Since some children may not have been initialized with state or valid actions
            if child.state == None:
                next_game_state , _ = from_state_action_to_state(self.emulator, tree.state.game_state, child.action)
                child.state = State.from_game_state(next_game_state)
                # Add the state and tree object to dictionary
                child.valid_actions = get_valid_actions(child.state.game_state)
            tree = child        

        # Now tree is assumed to be a leaf node
        # Check if the node has been traversed
        if tree.visit == 0:
            # Sometimes this happens
            if tree.state.game_state["table"].seats.players[0].hole_card == []:
                reward = 0
            else:
                reward = []
                # Instead of doing 1 rollout, we do many since we are limited on memory but not compute
                for rollout in range(self.num_rollouts):
                    reward.append(self.rollout(tree.state, self.emulator))

        else:
            # If node has been visited, expand the tree and perform rollout
            # NOTE: all children do not have state or valid actions after expansion
            tree.expand(tree.valid_actions)

            # Rollout on first child, other children will eventually get rolled out via UCB1
            action, child_tree = next(iter(tree.children.items()))

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

            if tree.state.game_state["table"].seats.players[0].hole_card == []:
                reward = 0
            else:
                reward = []
                # Instead of doing 1 rollout, we do many since we are limited on memory but not compute
                for rollout in range(self.num_rollouts):
                    reward.append(self.rollout(tree.state, self.emulator))

        # Do backpropogation up the tree
        if isinstance(reward, list):
            for r in reward:
                self.backup(tree, r)
        else:
            self.backup(tree, reward)

        if tree.player == "main":
            nodes = add_state_tree_to_external(nodes, tree)
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
            # NOTE:  (reward - tree.value)/tree.visit from POMCP Paper
            if tree.player == "opp":
                tree.value += (reward - tree.value)/tree.visit
            else:
                tree.value -= (reward - tree.value)/tree.visit
            
            tree = tree.parent
        
    def rollout(self, state: State, emulator: Emulator):
        emulator = copy.copy(emulator)
        cur_stack = state.game_state["table"].seats.players[0].stack
        end_game_state, events = emulator.run_until_round_finish(state.game_state)
        
        # How much the main player gained or lost
        reward = end_game_state["table"].seats.players[0].stack - cur_stack
        return reward

    def rollout_hand_eval(self, state: State, emulator: Emulator):
        main_hole_cards = state.game_state["table"].seats.players[0].hole_card
        opp_hole_cards = state.game_state["table"].seats.players[1].hole_card
        community_cards = state.game_state["table"].get_community_card()
        heuristic = self.hand_evaluator.eval_hand(main_hole_cards, community_cards) - self.hand_evaluator.eval_hand(opp_hole_cards, community_cards)
        reward = heuristic
        return reward

def is_round_finish(game_state):
    return game_state["street"] != Const.Street.FINISHED

if __name__ == '__main__':
    from pypokerengine.api.emulator import Emulator
    import json

    mcts = MCTS()
    nodes = mcts.search()
    time_out = mcts.timeout
    reinvigoration = mcts.reinvigoration
       
    # with open(f'search_tree_{time_out}_reinvigoration-{reinvigoration}.json', 'w') as f:
    with open(f'test.json', 'w') as f:
        json.dump(nodes, f, indent=4)