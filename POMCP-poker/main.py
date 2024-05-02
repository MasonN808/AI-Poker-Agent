from populate import POMCP

# Calculate policy in a loop
time = 0
history = []
pomcp = POMCP()
while time <= 100:
    time += 1
    action = pomcp.Search()
    # print(ab.tree.nodes[-1][:4])
    move_str = str(action_index_to_move(board, action))
    print(move_str)
    history.append(move_str)
    winner = board.apply_move(action_index_to_move(board, action))
    if winner is not None:
        print("Winner is", winner)
        break
    observation = board.to_fow_fen(board.side_to_move)  # choice(O)
    print(observation)
    print(board)
    pomcp.tree.prune_after_action(action, observation)
    pomcp.UpdateBelief(action, observation)
print(history)