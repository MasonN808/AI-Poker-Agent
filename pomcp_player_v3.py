import random
import math
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from pypokerengine.api.emulator import apply_action
from blackjack.utils import Observation, State
from pypokerengine.engine.round_manager import RoundManager
from pypokerengine.engine.hand_evaluator import hand_eval
from pypokerengine.utils.game_state_utils import restore_game_state

from randomplayer import RandomPlayer

class Particle:
    # TODO: Mason - we need to adjust our state and observation space by adding more features. I think we need information about
    # what the opposing agent has done already (e.g., call, fold, or raise). this should be available in round_state. We need this b/c
    # we need more partial information about the other player otherwise we can't generate a decent probability distribution of observations
    # over states.
    # TODO: mason - add call and raise cointer of opponnent to better infer the belief state.
    def __init__(self, obs: Observation, s: State):
        self.obs = obs
        self.s = s

    @classmethod
    def from_obs(cls, obs):
        """
        Samples a possible state from a uniform distribution given an observation
        """
        return cls(obs, obs.sample_state())
    
    @classmethod
    def from_s(cls, s):
        """
        Gets the observation from the state
        """
        return cls(s.get_observation(), s)

    

class SearchTree:
    def __init__(self, particles=None, action=None, visit=1, value=0):
        self.particles = particles if particles is not None else []
        self.action = action
        self.visit = visit
        self.value = value
        self.children = {}

    def expand(self, valid_actions):
        for action in valid_actions:
            self.children[action['action']] = SearchTree(action=action['action'])

    def ucb(self, child):
        log_div = math.log(self.visit) / child.visit
        return math.sqrt(log_div)

class POMCPPlayer(BasePokerPlayer):
    """
    Combines Monte-Carlo belief state updates with PO-UCT Algorithm
    """
    def __init__(self,
                 discount=0.8,
                 depth=0,
                 epsilon=1e-7,
                 explore=7,
                 n_particles=128,
                 reinvigoration=16):

        self.discount = discount
        self.depth = depth
        self.epsilon = epsilon
        self.explore = explore
        self.n_particles = n_particles
        self.reinvigoration = reinvigoration
        self.rollout_policy = RandomPlayer()
        self.tree = None # TODO: Need to adjust this for all the tree variables
        self.cur_node =
    
    def declare_action(self, valid_actions, hole_card, round_state):
        """
        Takes in a belief state and outputs an action
        """
        obs = Observation(hole_card, round_state)
        at_root = self.tree is None
        
        if at_root:
            tree = SearchTree()
            self.tree = tree
            self.tree.particles = [Particle.from_obs(obs) for _ in range(self.n_particles)]
        else:
            particles = [part for part in tree.particles if part.obs == obs]
            for _ in range(self.reinvigoration):
                # Use particle reinvigoration by adding particles with noise 
                # uniform sample from possible states given current observation
                particles.append(Particle.from_obs(obs))
            self.tree.particles = particles

        for _ in particles:
            # Take random sample from particles in 
            particle = random.sample(self.tree.particles, 1)[0]
            self.simulate(particle, tree, 0, valid_actions)

        child = max(self.tree.children.values(), key=lambda x: x.value)
        self.tree = child
        return child.best_action

    def simulate(self, particle, tree, depth, valid_actions):
        """
        Simulation performed using the PO-UCT Algorithm
        """
        if self.discount**depth < self.epsilon:
            return 0
        if depth >= self.max_depth:
            return 0  # Terminal depth reached

        # Since expand(), exands the tree for all actions we just need to check if the tree/node has children
        if not tree.children:
            tree.expand(valid_actions)
            # Lazy Evaluation
            discounted_return = self.rollout(particle, depth)
            # NOTE: Error in POMCP paper. This should be included
            tree.visit += 1
            tree.value = discounted_return
            return discounted_return

        # TODO: mason - which is more correct? Can valid actions be correct in all states? Same question as up top.
        actions = particle.s.actions()
        actions = valid_actions

        children = filter(lambda child: child.action in actions, tree.children)
        # Upper Confidence Bound (UCB) update
        child = max(children, key=lambda child: child.value + self.explore * tree.ucb(child))
        # Argmax
        action = child.action

        # Black-box step
        # new_s = particle.s.sample(action)
        game_state = restore_game_state(particle.s.round_state)
        new_game_s, messages = RoundManager.apply_action(game_state, action)
        # Convert next game state to State() object
        new_s = State.from_game_state(new_game_s, particle.s.hole_card_main, particle.s.hole_card_op)
        new_part = Particle.from_s(new_s)
        heuristic = hand_eval(new_s.hole_card_main, new_s.community_card) - hand_eval(new_s.hole_card_op, new_s.community_card)
        reward = heuristic + self.discount * self.simulate(new_part, child, depth + 1)
        tree.particles.append(new_part)

        # Tree updates
        tree.visit += 1
        child.visit += 1
        child.value += (reward - child.value) / child.visit
        return reward

    def rollout(self, particle, depth):
        if self.discount**depth < self.epsilon:
            return 0
        if depth >= self.max_depth:
            return 0  # Terminal depth reached
        
        # TODO: make rollout_policy random
        # Choose action using rollout policy (could be random)
        action = self.rollout_policy.policy(particle.obs, {})

        # Black-box step
        # new_s = particle.s.sample(action)
        game_state = restore_game_state(particle.s.round_state)
        # TODO: Double check that the next_game_s is the opponents turn to take an action
        new_game_opponent, messages = RoundManager.apply_action(game_state, action)
        # TODO: get valid actions of opponent
        # Calculate valid actions of opponent player
        new_round_opponent_state = new_game_opponent['round_state']

        # TODO: add conditional for possibility of game ending
        new_game_s, messages = RoundManager.apply_action(new_game_opponent, action_opponent)
        # Convert next game state to State() object
        new_s = State.from_game_state(new_game_s, particle.s.hole_card_main, particle.s.hole_card_op)
        new_part = Particle.from_s(new_s)
        heuristic = hand_eval(new_s.hole_card_main, new_s.community_card) - hand_eval(new_s.hole_card_op, new_s.community_card)
        reward = heuristic + self.discount * self.rollout(new_part, depth + 1)

        return reward  # Assuming this is the reward
    
    # def rollout(self, particle, depth):
    #     if self.discount**depth < self.epsilon:
    #         return 0
    #     if depth >= self.max_depth:
    #         return 0  # Terminal depth reached

    #     current_win_rate = estimate_hole_card_win_rate(nb_simulation=100, nb_player=2, hole_card=gen_cards(particle.hole_card), community_card=gen_cards(particle.community_card))
    #     return current_win_rate  # Assuming this is the reward
    
    # TODO: mason - A possible new way to perform ACTUAL rollouts
    # from pypokerengine.api.game import setup_config, start_poker
    # from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
    # from pypokerengine.players import BasePokerPlayer
    # import random

    # class RandomPlayer(BasePokerPlayer):
    #     def declare_action(self, valid_actions, hole_card, round_state):
    #         # Randomly choose an action from the valid ones
    #         action = random.choice(valid_actions)
    #         return action['action'], action['amount']

    # def rollout_simulation(starting_state):
    #     # Setup the configuration with your defined players
    #     config = setup_config(max_round=1, initial_stack=1000, small_blind_amount=10)
    #     config.register_player(name="player1", algorithm=RandomPlayer())
    #     config.register_player(name="player2", algorithm=RandomPlayer())

    #     # Start a poker game with the given configuration and state
    #     game_result = start_poker(config, verbose=0, initial_state=starting_state)
    #     return game_result

    # # Define a function to perform a rollout from the current state
    # def perform_rollout(current_state):
    #     result = rollout_simulation(current_state)
    #     # Evaluate the result and return the evaluation (e.g., chip difference)
    #     return result['players'][0]['stack'] - 1000  # Assuming player1's stack change as the outcome


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
    return POMCPPlayer()