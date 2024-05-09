import math
import sys

sys.path.insert(0, './')
from MCTS_poker.utils import State
sys.path.insert(0, './pypokerengine/api/')
import game
setup_config = game.setup_config
start_poker = game.start_poker
import time
from argparse import ArgumentParser


""" =========== *Remember to import your agent!!! =========== """
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from hand_eval_player import HeuristicPlayer
from MCTS_poker.mcts_player import MCTSPlayer
""" ========================================================= """

""" Example---To run testperf.py with random warrior AI against itself. 

$ python testperf.py -n1 "Random Warrior 1" -a1 RandomPlayer -n2 "Random Warrior 2" -a2 RandomPlayer
"""

def testperf(agent_name1, agent1, agent_name2, agent2):		

	# Init to play 500 games of 1000 rounds
	num_game = 500
	max_round = 1000
	initial_stack = 1000
	smallblind_amount = 10

	# Init pot of players
	agent1_pot = 0
	agent2_pot = 0

	# Setting configuration
	config = setup_config(max_round=max_round, initial_stack=initial_stack, small_blind_amount=smallblind_amount)
	

# 	==>> self.in_table: 14158
# ==>> self.not_in_table: 36163
	# Register players
	# These just dont work after the absraction on suits
	# config.register_player(name="MCTS", algorithm=MCTSPlayer(search_tree='search_tree_1000000_reinvigoration-1000__reinvigoration-5.json'))
	# config.register_player(name="MCTS", algorithm=MCTSPlayer(search_tree='search_tree_200000_reinvigoration-1000__reinvigoration-5.json'))
 
	# This runs well
	# config.register_player(name="MCTS", algorithm=MCTSPlayer(search_tree='search_tree_200000_reinvigoration-10_explore-100.json'))
	# This runs terribly
	# config.register_player(name="MCTS", algorithm=MCTSPlayer(search_tree='search_tree_200000_reinvigoration-10_explore-120.json'))

	# This runs about same as heuristic
	# config.register_player(name="MCTS", algorithm=MCTSPlayer(search_tree='search_tree_200000_reinvigoration-10_explore-100-belief_state_update.json'))

	# This performs terribly
	# config.register_player(name="MCTS", algorithm=MCTSPlayer(search_tree='search_tree_1000000_reinvigoration-20_explore-100-belief_state_update-bad.json'))

	# Performs to expectation
	# config.register_player(name="MCTS", algorithm=MCTSPlayer(search_tree='search_tree_200000_reinvigoration-10_explore-100-belief_state_update.json'))

	# Performs poorly
	# config.register_player(name="MCTS", algorithm=MCTSPlayer(search_tree='tree_200000_reinvigoration-1000_explore-100-n_particles-32.json'))

	# config.register_player(name="MCTS", algorithm=MCTSPlayer(search_tree='tree_10000_reinvigoration-1000_explore-100-n_particles-128.json'))
	# config.register_player(name="MCTS", algorithm=MCTSPlayer(search_tree='tree_10000_reinvigoration-10_explore-100-n_particles-8.json'))
	config.register_player(name="MCTS", algorithm=MCTSPlayer(search_tree='test.json'))
	config.register_player(name="Heuristic", algorithm=HeuristicPlayer())
	# config.register_player(name="MCTS", algorithm=MCTSPlayer())
	# config.register_player(name="Raised", algorithm=RaisedPlayer())
	# config.register_player(name="Raised", algorithm=RaisedPlayer())
	# config.register_player(name="Random", algorithm=RandomPlayer())
	# config.register_player(name="Random", algorithm=RandomPlayer())
	
	agent_1_wins = 0
	agent_2_wins = 0
	# Start playing num_game games
	for game in range(1, num_game+1):
		# print("Game number: ", game)
		game_result = start_poker(config, verbose=0)
		if game_result['players'][0]['stack'] > 1000:
			agent_1_wins += 1
		if game_result['players'][1]['stack'] > 1000:
			agent_2_wins += 1
		agent1_pot = agent1_pot + game_result['players'][0]['stack']
		agent2_pot = agent2_pot + game_result['players'][1]['stack']

	print(agent_1_wins)
	print(agent_2_wins)

	print("\n After playing {} games of {} rounds, the results are: ".format(num_game, max_round))
	# print("\n Agent 1's final pot: ", agent1_pot)
	print("\n " + agent_name1 + "'s final pot: ", agent1_pot)
	print("\n " + agent_name2 + "'s final pot: ", agent2_pot)

	# print("\n ", game_result)
	# print("\n Random player's final stack: ", game_result['players'][0]['stack'])
	# print("\n " + agent_name + "'s final stack: ", game_result['players'][1]['stack'])

	if (agent1_pot<agent2_pot):
		print("\n Congratulations! " + agent_name2 + " has won.")
	elif(agent1_pot>agent2_pot):
		print("\n Congratulations! " + agent_name1 + " has won.")
		# print("\n Random Player has won!")
	else:
		print("\n It's a draw!") 

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-n1', '--agent_name1', help="Name of agent 1", default="MCTS", type=str)
    parser.add_argument('-a1', '--agent1', help="Agent 1")    
    # parser.add_argument('-n2', '--agent_name2', help="Name of agent 2", default="RAISED", type=str)
    # parser.add_argument('-a2', '--agent2', help="Agent 2", default=RaisedPlayer())    
    parser.add_argument('-n2', '--agent_name2', help="Name of agent 2", default="RANDOM", type=str)
    parser.add_argument('-a2', '--agent2', help="Agent 2")    
    args = parser.parse_args()
    return args.agent_name1, args.agent1, args.agent_name2, args.agent2

if __name__ == '__main__':
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

	name1, agent1, name2, agent2 = parse_arguments()
	start = time.time()
	testperf(name1, agent1, name2, agent2)
	# testperf()
	end = time.time()

	print("\n Time taken to play: %.4f seconds" %(end-start))