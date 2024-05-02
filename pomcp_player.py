from pypokerengine.players import BasePokerPlayer
import random as rand
from pypokerengine.utils.game_state_utils import restore_game_state
import pprint

class POMCPPlayer(BasePokerPlayer):

  def declare_action(self, valid_actions, hole_card, round_state):
    # TODO: This is where pomcp policy will be
    # What exactly is stack? the players' amount or players amount betted or is that pot?
    # what is dealer_btn?
    # what is next_player?

    pp = pprint.PrettyPrinter(indent=2)
    print("------------ROUND_STATE(POMCP)--------")
    pp.pprint(round_state)
    print("------------HOLE_CARD----------")
    pp.pprint(hole_card)
    print("------------VALID_ACTIONS----------")
    pp.pprint(valid_actions)
    if len(valid_actions) < 3:
      exit()
    print("-------------------------------")
    game_state = restore_game_state(round_state)
    print(game_state)
    # print("-------PRINTS ALL POSSIBLE CARDS IN DECK--------")
    # game_state = restore_game_state(round_state)
    # pp.pprint([card.__str__() for card in game_state["table"].deck.deck])


    r = rand.random()
    if r <= 0.5:
      call_action_info = valid_actions[1]
    elif r<= 0.9 and len(valid_actions ) == 3:
      call_action_info = valid_actions[2]
    else:
      call_action_info = valid_actions[0]
    action = call_action_info["action"]
    print("-------PRINTS ALL POSSIBLE CARDS IN DECK--------")
    print(action)
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
  return POMCPPlayer()
