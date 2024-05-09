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
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


""" =========== *Remember to import your agent!!! =========== """
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from hand_eval_player import HeuristicPlayer
from MCTS_poker.mcts_player import MCTSPlayer
""" ========================================================= """

""" Example---To run testperf.py with random warrior AI against itself. 

$ python testperf.py -n1 "Random Warrior 1" -a1 RandomPlayer -n2 "Random Warrior 2" -a2 RandomPlayer
"""

def testperf_and_plot():		
	meta_agent_1_stacks = [] # Heuristic
	meta_agent_2_stacks = [] # Random
	meta_agent_3_stacks = [] # Raised
	meta_data = []
	for i in range(3):
		agent_i_stacks = []
		agent_j_stacks = []
		for j in range(10):
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
			
			# Register players
			if i == 0:
				config.register_player(name="Heuristic", algorithm=HeuristicPlayer())
				config.register_player(name="Random", algorithm=RandomPlayer())
			if i == 1:
				config.register_player(name="Heuristic", algorithm=HeuristicPlayer())
				config.register_player(name="Raised", algorithm=RaisedPlayer())
			if i == 2:
				config.register_player(name="Raised", algorithm=RaisedPlayer())
				config.register_player(name="Random", algorithm=RandomPlayer())
			

			# Start playing num_game games
			for game in range(1, num_game+1):
				print("Game number: ", game)
				game_result = start_poker(config, verbose=0)

				agent1_pot = agent1_pot + game_result['players'][0]['stack']
				agent2_pot = agent2_pot + game_result['players'][1]['stack']
			

			# For plotting purposes
			agent_i_stacks.append(agent1_pot)
			agent_j_stacks.append(agent2_pot)
			
		meta_data.append([agent_i_stacks, agent_j_stacks])

	fig, ax = plt.subplots(figsize=(12, 6))

    # Since you mentioned 3 groups of two whisker plots each
	data_groups = [meta_data[i] for i in range(3)]  # Assuming meta_data is as structured above

	colors = {'Heuristic': 'skyblue', 'Random': 'lightgreen', 'Raised': 'salmon'}
	group_labels = ['Heuristic vs. Random', 'Heuristic vs. Raised', 'Raised vs. Random']
	agent_labels = [['Heuristic', 'Random'], ['Heuristic', 'Raised'], ['Raised', 'Random']]
	xtick_locs = [1, 5, 9]  # X locations for the groups
	offset = 1.5  # Offset for boxes within the same group

	# Plot each group
	for idx, (group, agents) in enumerate(zip(meta_data, agent_labels)):
		positions = [xtick_locs[idx], xtick_locs[idx] + offset]
		box = ax.boxplot(group, positions=positions, widths=0.6, patch_artist=True)
		for patch, agent in zip(box['boxes'], agents):
			patch.set_facecolor(colors[agent])

	# Label formatting
	ax.set_xticks([x + offset / 2 for x in xtick_locs])
	ax.set_xticklabels(group_labels, fontsize=12)
	ax.set_ylabel('Post-Game Stack Size', fontsize=15)
	ax.set_xlabel('Agent Matchups', fontsize=15)
	ax.set_title('Post-Game Stack Size Comparisons Across Different Base Players', fontsize=18)

	# Grid lines for better readability
	ax.yaxis.grid(True)  # Horizontal grid lines
	ax.xaxis.grid(True, linestyle='--')  # Vertical grid lines

	# Set y-axis to scientific notation
	ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
	formatter = ScalarFormatter(useMathText=True)  # Use MathText for "x10^" formatting
	formatter.set_scientific(True)
	formatter.set_powerlimits((-1,1))
	ax.yaxis.set_major_formatter(formatter)


	# Create legend
	from matplotlib.patches import Patch
	legend_elements = [Patch(facecolor=colors[agent], label=agent) for agent in colors]
	ax.legend(handles=legend_elements, title='Agent Types', loc='upper right', framealpha=1)


	plt.savefig('heuristic_benchmark-v2.png')
	plt.show()


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-n1', '--agent_name1', help="Name of agent 1", default="MCTS", type=str)
    parser.add_argument('-a1', '--agent1', help="Agent 1", default=MCTSPlayer())    
    # parser.add_argument('-n2', '--agent_name2', help="Name of agent 2", default="RAISED", type=str)
    # parser.add_argument('-a2', '--agent2', help="Agent 2", default=RaisedPlayer())    
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
	testperf_and_plot()
	# testperf()
	end = time.time()

	print("\n Time taken to play: %.4f seconds" %(end-start))