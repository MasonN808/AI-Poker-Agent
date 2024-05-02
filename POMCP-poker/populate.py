import random
import math
import sys

sys.path.insert(0, './')
from pypokerengine.engine.table import Table
from pypokerengine.engine.seats import Seats
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
# from pypokerengine.api.emulator import apply_action
from blackjack.utils import Observation, State
from pypokerengine.engine.round_manager import RoundManager
# from pypokerengine.engine.hand_evaluator import eval_hand
from pypokerengine.utils.game_state_utils import restore_game_state
from pypokerengine.engine.poker_constants import PokerConstants as Const

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
    def __init__(self, obs, B=None, action=None, visit=1, value=0):
        self.particles = particles if particles is not None else []
        self.obs # Observation
        self.B = B if B is not None else [] # Belief states
        self.action = action
        self.visit = visit
        self.value = value
        self.children = {}
        self.actions = ['fold', 'call', 'raise'] # Default valid actions NOTE this will change in rounds

    def expand(self, valid_actions):
        for action in valid_actions:
            # TODO: overwrite possible actions for child node. Check when 
            self.children[action['action']] = SearchTree(action=action['action'])

    def ucb(self, child):
        log_div = math.log(self.visit) / child.visit
        return math.sqrt(log_div)

class POMCP():
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
        self.tree = SearchTree() # This is h
        self.emulator = Emulator()
        self.timeout = 1000
    
    # def search(self, valid_actions, hole_card, round_state):
    #     """
    #     Takes in a belief state and outputs an action
    #     """
    #     obs = Observation(hole_card, round_state)
    #     at_root = self.tree is None
        
    #     if at_root:
    #         tree = SearchTree()
    #         self.tree = tree
    #         self.tree.particles = [Particle.from_obs(obs) for _ in range(self.n_particles)]
    #     else:
    #         particles = [part for part in tree.particles if part.obs == obs]
    #         for _ in range(self.reinvigoration):
    #             # Use particle reinvigoration by adding particles with noise 
    #             # uniform sample from possible states given current observation
    #             particles.append(Particle.from_obs(obs))
    #         self.tree.particles = particles

    #     for _ in particles:
    #         # Take random sample from particles in 
    #         particle = random.sample(self.tree.particles, 1)[0]
    #         self.simulate(particle, tree, 0, valid_actions)

    #     child = max(self.tree.children.values(), key=lambda x: x.value)
    #     self.tree = child
    #     return child.best_action

    # Search module
    def search(self, state=None):
        # Repeat Simulations until timeout
        for _ in range(self.timeout):
            particles = self.tree.particles.copy()
            if particles == []:
                state = State.random_state()  # s ~ I(s_0=s)
            else:
                # TODO: Make this just the true state?
                state = random.choice(particles) # s ~ B(h)
            self.simulate(state, self.tree, 0)
        # Get best action
        # action, _ = self.SearchBest(-1, UseUCB=False)
        # return action

    def simulate(self, state, tree, depth):
        """
        Simulation performed using the PO-UCT Algorithm
        """
        if (self.discount**depth < self.epsilon) and  depth >= self.max_depth:
            return 0

        # If leaf node
        if not tree.children:
            tree.expand(tree.actions)
            # Lazy Evaluation
            discounted_return = self.rollout(tree, depth)
            # NOTE: Error in POMCP paper. This should be included
            tree.visit += 1
            tree.value = discounted_return
            return discounted_return

        # This is ha
        children = filter(lambda child: child.action in tree.actions, tree.children)
        # Upper Confidence Bound (UCB) update
        child = max(children, key=lambda child: child.value + self.explore * tree.ucb(child))
        # Argmax
        action = child.action

        # Black-box step
        game_state = restore_game_state(state.round_state)
        new_game_s, messages = RoundManager.apply_action(game_state, action)
        new_game_s, messages = RoundManager.apply_action(game_state, action)
        # Convert next game state to State() object
        new_s = State.from_game_state(new_game_s)
        new_part = Particle.from_s(state)
        # TODO: Edit Heursitic since rollout gives different rewards (could be off balanced)
        heuristic = eval_hand(new_s.hole_card_main, new_s.community_card) - eval_hand(new_s.hole_card_op, new_s.community_card)
        reward = heuristic + self.discount * self.simulate(new_part, child, depth + 1)
        tree.particles.append(part) # TODO: This should be 

        # Tree updates
        tree.visit += 1
        child.visit += 1
        child.value += (reward - child.value) / child.visit
        return reward

    def rollout(self, particle, depth):
        if (self.discount**depth < self.epsilon) and  depth >= self.max_depth:
            return 0

        # Black-box step
        game_state = restore_game_state(particle.s.round_state)
        end_game_state, events = self.emulator.run_until_round_finish(game_state)
        # reward = events[?] # TODO: Finish and figure out how to get value from game state (USE events)
        
        reward = end_game_state['winners']
        return reward  # Assuming this is the reward


def is_round_finish(game_state):
    return game_state["street"] != Const.Street.FINISHED

if __name__ == '__main__':
    from pypokerengine.api.emulator import Emulator
    import pprint
    
    num_player = 2
    max_round = 1000
    small_blind_amount = 10
    ante = 0 
    emulator = Emulator()
    emulator.set_game_rule(num_player, max_round, small_blind_amount, ante)
    # emulator.set_game_rule(player_num=3, max_round=10, small_blind_amount=10, ante_amount=0)

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

    pp = pprint.PrettyPrinter(indent=2)
    # print("------------ROUND_STATE(POMCP)--------")
    # pp.pprint(round_state)
    # print("------------HOLE_CARD----------")
    # pp.pprint(hole_card)
    # print("------------VALID_ACTIONS----------")
    # pp.pprint(valid_actions)
    # if len(valid_actions) < 3:
    #     exit()
    
    # print("-------------------------------")
    # game_state = restore_game_state(round_state)
    # print(game_state)

    for _ in range(1):
        initial_state = emulator.generate_initial_game_state(players_info)
        print("------------INITIAL_STATE--------")
        pp.pprint(initial_state)
        pp.pprint(len(initial_state["table"].seats.players[0].hole_card))
        game_state, events = emulator.start_new_round(initial_state)
        print("------------GAME_STATE----------")
        pp.pprint(game_state)
        pp.pprint([str(i) for i in game_state["table"].get_community_card()])
        pp.pprint([str(i) for i in game_state["table"].seats.players[0].hole_card])
        pp.pprint([str(i) for i in game_state["table"].seats.players[1].hole_card])
        print("------------RAISED----------")
        game_state, msg = RoundManager.apply_action(game_state, "call")
        game_state, msg = RoundManager.apply_action(game_state, "call")
        # pp.pprint(game_state)
        # pp.pprint(msg)
        pp.pprint([str(i) for i in game_state["table"].get_community_card()])
        pp.pprint([str(i) for i in game_state["table"].seats.players[0].hole_card])
        pp.pprint([str(i) for i in game_state["table"].seats.players[1].hole_card])
        print("------------EVENTS----------")
        # pp.pprint(events)
        end_game_state, events = emulator.run_until_round_finish(game_state)
        pp.pprint([str(i) for i in end_game_state["table"].seats.players[0].hole_card])
        pp.pprint([str(i) for i in end_game_state["table"].seats.players[1].hole_card])

        # RoundManager.apply_action(game_state, )

