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
from MCTS_poker.utils import State, add_state_tree_to_external, from_state_action_to_state, get_valid_actions, sort_cards, sort_cards_card_obj
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

        self.my_pot_portion = 0

    def expand(self, valid_actions: list[str]):
        assert valid_actions != None, "valid actions should not be none"
        for action in valid_actions:
            if self.player == "main":
                player = "opp"
            else:
                player = "main"

            # reward = 
            # if action == "fold":
                # self.children[action] = SearchTree(player=player, action=action, parent=self)
            # else:
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
                 explore=100,
                 n_particles=64):

        self.explore = explore
        self.n_particles = n_particles
        self.emulator = None
        self.hand_evaluator = HandEvaluator()

        self.num_rollouts = 1
        # self.timeout = 5000
        # self.timeout = 50000
        self.timeout = 200_000
        # self.timeout = 600_000
        # self.timeout = 4_500_000
        # self.timeout = 1_000_000
        # self.timeout = 10_000_000
        # self.timeout = 50_000_000
        self.reinvigoration = 10

    # Search module
    def search(self, state=None):
        global nodes

        for t in tqdm(range(self.timeout), desc='Progress'):
            if state == None:
                # Sample an initial state (observation) and get the initialized pypoker emulator
                state_instance = State()
                state, self.emulator = state_instance.random_state()  # s ~ I(s_0=s)
            # print(f"==>> state: {state.state_info}")
            assert not is_round_finish(state.game_state)
            if state.game_state['next_player'] == 1:
                player = "main"
            else:
                player = "opp"

            # Check if state info is a key in nodes to avoid creating unnecessary new tree objects
            if state.state_info in nodes and player == "main":
                sorted_card_str = sort_cards(state.state_info)
                is_unique = True
                # Loop through all current belief states
                sorted_prunned_opp_hole_cards_tree = sort_cards_card_obj(state.game_state["table"].seats.players[1].hole_card)
                for nodes_tree in nodes[sorted_card_str]:
                    # print(len(nodes[sorted_card_str]))
                    # Sort the opponnets hole cards and remove the suits
                    # print([card.__str__() for card in nodes_tree.state.game_state["table"].seats.players[1].hole_card])
                    # print([card for card in sort_cards_card_obj(nodes_tree.state.game_state["table"].seats.players[1].hole_card)])
                    sorted_prunned_opp_hole_cards_nodes = sort_cards_card_obj(nodes_tree.state.game_state["table"].seats.players[1].hole_card)
                    # If the opponnet has the same hole cards as a tree in the nodes tree then break and dont add the tree to the trees list
                    # TODO: this may be eroneus due to the fact that community cards may never be allocated during the behinning rounds but it may be trivial for our case
                    if (sorted_prunned_opp_hole_cards_tree == sorted_prunned_opp_hole_cards_nodes):
                        # print("IS UNIQUE")
                        is_unique = False
                        break

                if is_unique:
                    tree = SearchTree(player=player, state=state, action=None, parent=None)
                    self.simulate(tree)

                # if state.game_state


                belief_trees = nodes[state.state_info]

                traversed_belief_trees = []
                if len(belief_trees) > 0:
                    # print(len(belief_trees))
                    i = 0
                    # Upperboud the while loop to prevent hangs
                    while belief_trees and i < self.n_particles:
                        i += 1
                        # Select and remove a random element from belief_trees

                        random_index = random.randint(0, len(belief_trees) - 1)
                        random_tree = belief_trees[random_index]

                        self.simulate(random_tree)

                        traversed_belief_trees.append(random_tree)
                        belief_trees.remove(random_tree)

                    # Reappend the traversed trees
                    belief_trees = belief_trees + traversed_belief_trees
                    
                    nodes[state.state_info] = belief_trees

                    # Reset to save memory
                    traversed_belief_trees = []

                    # Sample from the start state every n simulations
                    if (t + 1) % self.reinvigoration == 0:
                        state = None 
                    # Skip outside simulate
                    continue
                else:
                    # There are no belief states so just make a new tree
                    tree = SearchTree(player=player, state=state, action=None, parent=None)
            else:
                # There are no entries of state info in nodes so do this
                tree = SearchTree(player=player, state=state, action=None, parent=None)

            self.simulate(tree)

            # Sample from the start state every n simulations
            if (t + 1) % self.reinvigoration == 0:
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
                if tree.children == {} or tree.visit < 2:
                    pass
                else:
                    action = max(tree.children.values(), key=lambda child: child.value).action
                    optimal_actions.append(action)

            # If none of the trees have children rely on heursitic policy
            if optimal_actions == []:
                pass
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

    def simulate(self, tree):
        """
        Simulation performed using the UCT Algorithm
        """
        global nodes

        assert tree.state != None, "State is None"
        if tree.valid_actions == None:
            tree.valid_actions = get_valid_actions(tree.state.game_state)

        # Keep going down tree until a node with no children is found
        while tree.children:
            # Replace current node with the child that maximized UCB1(s) value
            child = max(tree.children.values(), key=lambda child: child.value + self.explore * tree.ucb(child) if child.action != "fold" else -100000)
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
            # if tree.state.game_state["table"].seats.players[0].hole_card == []:
            #     reward = 0
            # else:
            reward = self.rollout(tree.state, self.emulator)

        else:
            if is_round_finish(tree.state.game_state):
                # print(tree.parent.state.game_state)
                # print(tree.state.game_state)
                # print(tree.state.game_state["table"].seats.players[0].stack)
                # print(tree.state.game_state["table"].seats.players[1].stack)
                cur_stack = 1000
                # print(f"==>> cur_stack: {cur_stack}")
                # reward = self.rollout(tree.state, self.emulator)
                # print(tree.state.game_state["table"].seats.players[0].name)
                # end_game_state, events = self.emulator.apply_action(tree.parent.state.game_state, tree.state.action)
        
                # How much the main player gained or lost
                reward = tree.state.game_state["table"].seats.players[0].stack - cur_stack
                # print(tree.state.game_state["table"].seats.players[1].hole_card)
                # print(reward)
                # exit()
            else:
                # If node has been visited, expand the tree and perform rollout
                # NOTE: all children do not have state or valid actions after expansion
                tree.expand(tree.valid_actions)

                # Rollout on first child, other children will eventually get rolled out via UCB1
                action, child_tree = next(iter(tree.children.items()))

                # Need to reset the players stack to prevent game from ending
                # TODO: Idk if this is right
                # tree.state.game_state["table"].seats.players[0].stack = 1000
                # tree.state.game_state["table"].seats.players[1].stack = 1000
                # Extract resulting state for child node after performing action from parent node
                next_game_state , messages = from_state_action_to_state(self.emulator, tree.state.game_state, action)
                # Check if next_game_state is end of round
                if is_round_finish(next_game_state):
                    cur_stack = 1000

                    # How much the main player gained or lost
                    reward = tree.state.game_state["table"].seats.players[0].stack - cur_stack
                    # return
                else:
                    tree = child_tree
                    tree.state = State.from_game_state(next_game_state)
                    tree.valid_actions = get_valid_actions(next_game_state)

                    reward = self.rollout(tree.state, self.emulator)

        # Do backpropogation up the tree
        self.backup(tree, reward)

        if is_round_finish(tree.state.game_state):
            return

        # print(tree.state.game_state)
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
    return game_state["street"] == Const.Street.FINISHED

if __name__ == '__main__':
    from pypokerengine.api.emulator import Emulator
    import json

    mcts = MCTS()
    state_actions = mcts.search()
    time_out = mcts.timeout
    reinvigoration = mcts.reinvigoration
    n_particles = mcts.n_particles
    explore = mcts.explore
       
    with open(f'tree_{time_out}_reinvigoration-{reinvigoration}_explore-{explore}-n_particles-{n_particles}.json', 'w') as f:
    # with open(f'test.json', 'w') as f:
        json.dump(state_actions, f, indent=4)