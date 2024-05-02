from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.game_state_utils import restore_game_state

import random as rand
import pprint

class RandomPlayer(BasePokerPlayer):

  def __init__(self, uuid=None):
    super().__init__()
    self.uuid = uuid

  def declare_action(self, valid_actions, hole_card, round_state):
    # valid_actions format => [raise_action_pp = pprint.PrettyPrinter(indent=2)
    pp = pprint.PrettyPrinter(indent=2)
    # print("------------ROUND_STATE(RANDOM)--------")
    # pp.pprint(self.uuid)
    # print("------------HOLE_CARD----------")
    # pp.pprint(hole_card)
    # game_state = restore_game_state(round_state)
    # # pp.pprint([card.__str__() for card in game_state["table"].deck.deck])
    # pp.pprint([str(i) for i in game_state["table"].seats.players[0].hole_card])
    # pp.pprint([str(i) for i in game_state["table"].seats.players[1].hole_card])
    # print("------------VALID_ACTIONS----------")
    # pp.pprint(valid_actions)
    # print("-------------------------------")
    r = rand.random()
    if r <= 0.5:
      call_action_info = valid_actions[1]
    elif r<= 0.9 and len(valid_actions ) == 3:
      call_action_info = valid_actions[2]
    else:
      call_action_info = valid_actions[0]
    action = call_action_info["action"]
    return action  # action returned here is sent to the poker engine

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
  return RandomPlayer()
