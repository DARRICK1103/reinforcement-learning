import numpy as np
import pandas as pd


State = 5
Actions = ['left', 'right']
Epsilon = 0.9
Alpha = 0.1
Gamma = 0.9
Time = 0.3

qtable = pd.DataFrame(np.zeros((State, len(Actions))), columns=Actions, )

def action(S, q_table):
    state_actions = q_table.iloc[S, :]
    if (np.random.uniform() > Epsilon) or ((state_actions == 0).all()):
        action_move = np.random.choice(Actions)
    else:  # act greedy
        action_move = state_actions.idxmax()
    return action_move

def update_move(S, A):
    if A == 'right':  # move right
        if S == State - 2:  # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:  # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R

def update_env(curr_position, Action):

    print('Current Location: ' + str(curr_position) + '         Move: ' + Action, end='\n')


if __name__ == "__main__":
    for episode in range(1, 16):
        count_step = 0
        curr_position = 0
        is_done = False
        print('\nEpisode ' + str(episode))
        while not is_done:

            A = action(curr_position, qtable)
            next, R = update_move(curr_position, A)  # take action & get next state and reward
            q_predict = qtable.loc[curr_position, A]
            if next != 'terminal':
                q_target = R + Gamma * qtable.iloc[next, :].max()  # next state is not terminal
            else:
                q_target = R  # next state is terminal
                is_done = True  # terminate this episode

            qtable.loc[curr_position, A] += Alpha * (q_target - q_predict)  # update

            update_env(curr_position, A)
            curr_position = next  # move to next state
            count_step += 1


    print('\r\nQ-table:\n')
    print(qtable)
