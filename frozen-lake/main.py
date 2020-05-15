import gym
import argparse
from agents.sarsa import SARSA
import tensorflow as tf
import os


def main(agent_type, episodes, exp_name):
    logdir = os.path.join("logs", exp_name)
    os.makedirs(logdir, exist_ok=True)
    writer = tf.summary.create_file_writer(logdir)
    env = gym.make('FrozenLake-v0', is_slippery=False)
    states = env.observation_space.n
    actions = env.action_space.n
    if agent_type == "SARSA":
        agent = SARSA(states=states, actions=actions)
    else:
        raise NotImplementedError

    average = 0
    step = 0
    for ep in range(episodes):
        s_curr = env.reset()
        a_curr = agent.get_action(s_curr)
        done = False
        while not done:
            env.render()
            s_next, r, done, _ = env.step(a_curr)
            if r == 0 and done:
                r = -1
                with writer.as_default():
                    tf.summary.scalar("reward", r, step)
            if done and r > 0:
                print("################Goal Reached###################", r)
                with writer.as_default():
                    tf.summary.scalar("reward", r, step)


            a_next = agent.get_action(s_next)
            agent.train(s_curr, a_curr, r, s_next, a_next)
            s_curr = s_next
            a_curr = a_next
            step += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=str, default="SARSA", help="agent type")
    ap.add_argument("--exp_name", type=str, default="SARSA", help="exp_name")
    ap.add_argument("--episodes", type=int, default=100000, help="number of episodes to run")
    args = vars(ap.parse_args())
    main(agent_type=args["agent"], episodes=args["episodes"], exp_name=args["exp_name"])
