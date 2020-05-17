import numpy as np
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from collections import namedtuple, deque

sample = namedtuple('sample', ['s_curr', 'a_curr', 'reward', 's_next', 'done'])


class DQN:
    def __init__(self, states, actions):
        self.replay_size = 1024
        self.experience_replay = deque(maxlen=self.replay_size)
        self.epsilon = 0.1
        self.min_epsilon = 0.01
        self.actions = actions
        self.states = states
        self.lr = 0.001
        self.gamma = 0.99
        self.batch_size = 64
        self.batch_indices = np.array([i for i in range(64)])
        # self.batch_indices = self.batch_indices[:, np.newaxis]
        self.target_network = self.define_model()
        self.main_network = self.define_model()

    def define_model(self):
        state = Input(self.states)
        x = Dense(16, kernel_initializer='he_uniform')(state)
        x = Activation("relu")(x)
        x = Dense(24, kernel_initializer='he_uniform')(x)
        x = Activation("relu")(x)
        q = Dense(self.actions, kernel_initializer='he_uniform', kernel_regularizer="l2")(x)
        model = Model(state, q)
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        return model

    def get_action(self, state):
        decision = np.random.rand()
        if decision < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            q = self.target_network.predict(state)
            action = np.argmax(q)
        return action

    def train(self, x_batch):
        s_currs = np.zeros((self.batch_size, self.states))
        a_currs = np.zeros((self.batch_size, 1))
        r = np.zeros((self.batch_size, 1))
        s_nexts = np.zeros((self.batch_size, self.states))
        dones = np.zeros((self.batch_size,))

        for batch in range(self.batch_size):
            s_currs[batch] = x_batch[batch].s_curr
            a_currs[batch] = x_batch[batch].a_curr
            r[batch] = x_batch[batch].reward
            s_nexts[batch] = x_batch[batch].s_next
            dones[batch] = x_batch[batch].done

        target = self.main_network.predict(s_currs)
        max_qs = np.amax(self.target_network.predict(s_nexts), 1)
        max_qs = max_qs[..., np.newaxis]
        a_indices = a_currs.astype(np.int)
        target[self.batch_indices, np.squeeze(a_indices)] = np.squeeze(r + self.gamma * (max_qs))

        done_indices = np.argwhere(dones)
        if done_indices.shape[0] > 0:
            done_indices = np.squeeze(np.argwhere(dones))
            target[done_indices, np.squeeze(a_indices[done_indices])] = np.squeeze(r[done_indices])

        self.main_network.train_on_batch(s_currs, target)

    def update_weights(self):
        self.target_network.set_weights(self.main_network.get_weights())
