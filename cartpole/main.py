import gym
import argparse
import tensorflow as tf
import os
from agents.dqn import DQN
import numpy as np
from collections import namedtuple
import random




def main(agent_type, episodes, exp_name):
    logdir = os.path.join("logs", exp_name)
    os.makedirs(logdir, exist_ok=True)
    writer = tf.summary.create_file_writer(logdir)
    env = gym.make('CartPole-v1')
    states = env.observation_space.shape[0] #shape returns a tuple
    actions = env.action_space.n
    if agent_type == "DQN":
        agent = DQN(states=states, actions=actions)
    else:
        raise NotImplementedError
    warmup_ep = 0
    for ep in range(episodes):
        s_curr = env.reset()
        s_curr = np.reshape(s_curr, (1, states))
        done = False
        score = 0
        agent.update_weights()
        if agent.epsilon > agent.min_epsilon:
            agent.epsilon *= 0.99
        while not done:
            # env.render()
            a_curr = agent.get_action(s_curr)
            s_next, r, done, _ = env.step(a_curr)
            s_next = np.reshape(s_next, (1, states))
            r = r if not done or r > 499 else -100
            sample = namedtuple('sample', ['s_curr', 'a_curr', 'reward', 's_next', 'done'])

            sample.s_curr = s_curr
            sample.a_curr = a_curr
            sample.reward = r
            sample.s_next = s_next
            sample.done = done

            if len(agent.experience_replay) < agent.replay_size:
                agent.experience_replay.append(sample)
                s_curr = s_next
                continue
            else:
                agent.experience_replay.append(sample)
                x_batch = random.sample(agent.experience_replay, agent.batch_size)
                agent.train(x_batch)

            score += r

            s_curr = s_next

            if done:
                score = score if score == 500 else score + 100
                print(f"ep:{ep-warmup_ep}:################Goal Reached###################", score)

                with writer.as_default():
                    tf.summary.scalar("reward", r, ep)
                    tf.summary.scalar("score", score, ep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=str, default="DQN", help="agent type")
    ap.add_argument("--exp_name", type=str, default="DQN_final", help="exp_name")
    ap.add_argument("--episodes", type=int, default=3000, help="number of episodes to run")
    args = vars(ap.parse_args())
    main(agent_type=args["agent"], episodes=args["episodes"], exp_name=args["exp_name"])
