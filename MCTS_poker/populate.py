from collections import Counter
import copy
import random
import math
import sys

sys.path.insert(0, './')
# from pypokerengine.api.emulator import apply_action
from MCTS_poker.utils import State, add_state_tree_to_external, from_state_action_to_state, get_valid_actions, sort_cards, sort_cards_card_obj
# from pypokerengine.engine.hand_evaluator import eval_hand
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
            #     self.children[action] = SearchTree(player=player, action=action, parent=self)
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
                 explore=200,
                 n_particles=91): # Max is 91 in theory

        self.explore = explore
        self.n_particles = n_particles
        self.emulator = None
        self.hand_evaluator = HandEvaluator()

        # self.timeout = 200_000
        # self.timeout = 50000
        # self.timeout = 500_000
        # self.timeout = 200_000
        # self.timeout = 600_000
        # self.timeout = 4_500_000
        self.timeout = 1_000_000
        # self.timeout = 10_000_000
        # self.timeout = 50_000_000
        self.reinvigoration = 20000

    # Search module
    def search(self, state=None):
        global nodes

        for t in tqdm(range(self.timeout), desc='Progress'):
            if state == None:
                # Sample an initial state (observation) and get the initialized pypoker emulator
                state_instance = State()
                state, self.emulator = state_instance.random_state()  # s ~ I(s_0=s)
            else:
                state_instance = State()
                state, self.emulator = state_instance.random_state(state)  # s ~ B(s_0=s)
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
                    # Sort the opponnets hole cards and remove the suits
                    sorted_prunned_opp_hole_cards_nodes = sort_cards_card_obj(nodes_tree.state.game_state["table"].seats.players[1].hole_card)
                    # If the opponnet has the same hole cards as a tree in the nodes tree then break and dont add the tree to the trees list
                    # Check if the state is already in the belief states
                    if (sorted_prunned_opp_hole_cards_tree == sorted_prunned_opp_hole_cards_nodes):
                        is_unique = False
                        break

                if is_unique:
                    tree = SearchTree(player=player, state=state, action=None, parent=None)
                    self.simulate(tree)

                    # Sample from the start state every n simulations
                    # if (t + 1) % self.reinvigoration == 0:
                    #     state = None 
                    # Skip outside simulate
                    # continue

                # if state.game_state


                belief_trees = nodes[state.state_info]
                # print(len(belief_trees))

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
                if tree.children == {}:
                    pass
                else:
                    if tree.player == "main":
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
                # print(key)
                # print(max_count)
                

                most_common_actions = [element for element, count in counter.items() if count == max_count]
                # print(most_common_actions)

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
            # Do not traverse the fold node since the result if deterministic given the game tree
            valid_children = {key: child for key, child in tree.children.items() if child.action != "fold"}
            if tree.player == "main":
                # Exclude children with 'fold' action or use float('-inf') as a deterrent
                if valid_children:
                    child = max(valid_children.values(), key=lambda child: child.value + self.explore * tree.ucb(child))
            else:
                if valid_children:
                    child = min(valid_children.values(), key=lambda child: child.value + self.explore * tree.ucb(child))

            # print(child.value)
            # Since some children may not have been initialized with state or valid actions
            if child.state == None:
                next_game_state , _ = from_state_action_to_state(self.emulator, tree.state.game_state, child.action)
                child.state = State.from_game_state(next_game_state)
                # Add the state and tree object to dictionary
                child.valid_actions = get_valid_actions(child.state.game_state)
            # print(f"==>> child.value: {child.state.state_info}-{child.action}-{child.value} ")
            tree = child        

        # Now tree is assumed to be a leaf node
        # Check if the node has been traversed
        if tree.visit == 0:
            reward = self.rollout(tree.state, self.emulator)

        else:
            # print(tree.visit)
            if is_round_finish(tree.state.game_state):
                cur_stack = 1000
                # How much the main player gained or lost
                reward = tree.state.game_state["table"].seats.players[0].stack - cur_stack
            else:
                # If node has been visited, expand the tree and perform rollout
                # NOTE: all children do not have state or valid actions after expansion
                tree.expand(tree.valid_actions)

                # Rollout on first child, other children will eventually get rolled out via UCB1
                action, child_tree = next(iter(tree.children.items()))

                # Extract resulting state for child node after performing action from parent node
                next_game_state , messages = from_state_action_to_state(self.emulator, tree.state.game_state, action)
                # Check if next_game_state is end of round
                if is_round_finish(next_game_state):
                    cur_stack = 1000
                    # How much the main player gained or lost
                    reward = next_game_state.game_state["table"].seats.players[0].stack - cur_stack
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
            # if tree.player == "opp":
            #     tree.value = tree.value + reward
            # else:
            #     tree.value = tree.value - reward
            if tree.player == "opp":
                tree.value = tree.value + (reward - tree.value)/tree.visit
            else:
                tree.value = tree.value - (reward - tree.value)/tree.visit
            
            tree = tree.parent
        
    def rollout(self, state: State, emulator: Emulator):
        emulator = copy.copy(emulator)
        # cur_stack = state.game_state["table"].seats.players[0].stack
        cur_stack = 1000
        # print(cur_stack)
        # cur_stack = 1000
        end_game_state, events = emulator.run_until_round_finish(state.game_state)
        
        # How much the main player gained or lost
        # This is in range of [-320, 320]
        reward = end_game_state["table"].seats.players[0].stack - cur_stack

        # print(reward)
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
       
    with open(f'new-tree_{time_out}_reinvigoration-{reinvigoration}_explore-{explore}-n_particles-{n_particles}.json', 'w') as f:
    # with open(f'test.json', 'w') as f:
        json.dump(state_actions, f, indent=4)