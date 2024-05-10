from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.card import Card

class HeuristicPlayer(BasePokerPlayer):
    def __init__(self, epsilon=52000):
        super().__init__()
        self.hand_evaluator = HandEvaluator()
        self.epsilon = epsilon


    def declare_action(self, valid_actions, hole_card, round_state):
        community_cards = [Card.from_str(str_card) for str_card in round_state["community_card"]]
        hole_cards = [Card.from_str(str_card) for str_card in hole_card]

        heuristic = self.hand_evaluator.eval_hand(hole_cards, community_cards)
        reward = heuristic

        if reward > self.epsilon:
            for i in valid_actions:
                if i["action"] == "raise":
                    action = i["action"]
                    return action
            action = valid_actions[1]["action"]
            return action
        # Just fold
        else:
            for i in valid_actions:
                if i["action"] == "fold":
                    action = i["action"]
                    return action
        
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
    return HeuristicPlayer()
