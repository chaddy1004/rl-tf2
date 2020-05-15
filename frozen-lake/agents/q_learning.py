import numpy as np


class QLearning:
    def __init__(self, states, actions):
        self.epsilon = 0.2
        self.actions = actions
        self.lr = 0.01
        self.gamma = 0.9
        self.q_table = np.zeros((states, actions))

    def get_action(self, state):
        decision = np.random.rand()
        if decision < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            q = self.q_table[state]
            action = np.argmax(q)
        return action

    def train(self, s_curr, a_curr, r, s_next, a_next):
        q_curr = self.q_table[s_curr, a_curr]
        print(self.q_table[s_curr])
        q_new = r + self.gamma * np.max(self.q_table[s_next])
        self.q_table[s_curr, a_curr] = q_curr + self.lr * (q_new - q_curr)
