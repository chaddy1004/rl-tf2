import numpy as np


class SARSA:
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
        q_next = self.q_table[s_next, a_next]
        print(self.q_table[s_curr])
        self.q_table[s_curr, a_curr] = q_curr + self.lr * (r + (self.gamma * q_next) - q_next)
