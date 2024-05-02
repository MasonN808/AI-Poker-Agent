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
from MCTS_poker.mcts_player import MCTSPlayer
""" ========================================================= """

""" Example---To run testperf.py with random warrior AI against itself. 

$ python testperf.py -n1 "Random Warrior 1" -a1 RandomPlayer -n2 "Random Warrior 2" -a2 RandomPlayer
"""

def testperf(agent_name1, agent1, agent_name2, agent2):		

	# Init to play 500 games of 1000 rounds
	num_game = 20
	max_round = 20
	initial_stack = 1000
	smallblind_amount = 10

	# Init pot of players
	agent1_pot = 0
	agent2_pot = 0

	# Setting configuration
	config = setup_config(max_round=max_round, initial_stack=initial_stack, small_blind_amount=smallblind_amount)
	
	# Register players
	config.register_player(name=agent_name1, algorithm=MCTSPlayer())
	config.register_player(name=agent_name2, algorithm=RandomPlayer())
	# config.register_player(name=agent_name1, algorithm=agent1())
	# config.register_player(name=agent_name2, algorithm=agent2())
	

	# Start playing num_game games
	for game in range(1, num_game+1):
		print("Game number: ", game)
		game_result = start_poker(config, verbose=0)
		agent1_pot = agent1_pot + game_result['players'][0]['stack']
		agent2_pot = agent2_pot + game_result['players'][1]['stack']

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
    parser.add_argument('-a1', '--agent1', help="Agent 1", default=MCTSPlayer())    
    parser.add_argument('-n2', '--agent_name2', help="Name of agent 2", default="RANDOM", type=str)
    parser.add_argument('-a2', '--agent2', help="Agent 2", default=RandomPlayer())    
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
	end = time.time()

	print("\n Time taken to play: %.4f seconds" %(end-start))