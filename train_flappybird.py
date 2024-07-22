
import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

from tqdm import trange 

from collections import namedtuple, deque
import matplotlib.pyplot as plt
from demonstrations import Demonstrations as Demos
import os
import imageio
from PIL import Image, ImageDraw
import heapq
from environments.flappy_bird import FlappyBirdEnv
import math

env = FlappyBirdEnv #Make a flappy bird env  then import and change this to work
env_name = "flappy_bird"
experiment_name = "PlayData"
n_actions = 2   
n_observations = 12 
experiment_trials = 1
num_demo_trials = [0] 

read_demo_filename = "BestDemo" 
write_demo_filename = "WatchedDemoLearning" 
read_policy_filename = "FlappyBirdOptimalPolicy"
write_policy_filename = "FlappyBirdOptimalPolicy" 
read_oracle_filename = "FlappyBirdOptimalOracle"
write_oracle_filename = "FlappyBirdOptimalOracle"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256
LR = 1e-4
EPISODES = 10000
TARGET_SCORE = 200    # early training stop at avg score of last 100 episodes
GAMMA = 0.99            # discount factor
MEMORY_SIZE = 100000    # max memory buffer size
LEARN_STEP = 2          # how often to learn
TAU = 0.001             # for soft update of target parameters
SAVE_CHKPT = True      # save trained network .pth file


#Q NETWORK ARCHITECTURE
class DeepQNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DeepQNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
            )

    def forward(self, x):
        return self.fc(x)

#DQN ALGORITHM
class DQN():
    def __init__(self, n_observations, n_actions, batch_size=64, lr=1e-4, gamma=0.99, mem_size=int(1e5), learn_step=5, tau=1e-3):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_step = learn_step
        self.tau = tau

        self.policy_net = DeepQNetwork(n_observations, n_actions).to(device)
        self.target_net = DeepQNetwork(n_observations, n_actions).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # REplay Buffer
        self.memory = PrioritizedReplayBuffer(n_actions, mem_size, batch_size)
        self.counter = 0

    def chooseAction(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()

        # epsilon-greedy
        if random.random() < epsilon:
            action = random.choice(np.arange(self.n_actions))
        else:
            action = np.argmax(action_values.cpu().data.numpy())

        return action

    def remember(self, state, action, reward, next_state, done, priority = None):
        if priority is None:
            td_error = reward + self.gamma * (self.target_net((torch.from_numpy(next_state).float().unsqueeze(0).to(device))).squeeze()[action])  - self.policy_net(torch.from_numpy(state).float().unsqueeze(0).to(device)).squeeze()[action]
            priority = td_error.item()

        self.counter += 1
        if self.counter % self.learn_step == 0:
            if len(self.memory) >= self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

        if priority is None:
            priority=10

        self.memory.add(state, action, reward, next_state, done, priority)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        q_target = self.target_net(next_states).detach().max(axis=1)[0].unsqueeze(1)
        y_j = rewards + self.gamma * q_target * (1 - dones)  
        q_eval = self.policy_net(states).gather(1, actions)

        #backpropogation
        loss = self.criterion(q_eval, y_j)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #Update Target Policy
        self.softUpdate()



    def softUpdate(self):
        for eval_param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau*eval_param.data + (1.0-self.tau)*target_param.data)
    

#PRIORITIZED EXPERIENCE REPLAY
class PrioritizedReplayBuffer():
    def __init__(self, n_actions, memory_size, batch_size):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.memory = []
        self.capacity = memory_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.counter = 0

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done, priority):
        if priority is None:
            priority = 0
        e = self.experience(state, action, reward, next_state, done)
        exp = (priority, self.counter, e)
        self.counter += 1

        if self.__len__() >= self.capacity-1:
            try:
                heapq.heappop(self.memory)
            except:
                # print(self.memory[0], exp)
                heapq.heapify(self.memory)
                self.memory.pop(0)

        self.memory.append(exp)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        _, _, experiences = zip(*experiences)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    

#EVALUATE POLICY
def evaluatePolicy(env, agent, loop=3):
    wins=0
    for i in range(loop):
        state  = env.reset()[0]
        for idx_step in range(10000):
            action = agent.chooseAction(state, epsilon=0)
            state, reward, done, info, _ = env.step(action)
            if done:
                if wins:
                    wins += 1
                break
    env.close()

    return wins/loop

def train(env, agent, n_episodes=1000, max_steps=3000, eps_start=1.0, eps_decay=0.999, eps_end=0.01, target=200, chkpt=False, demo_load=0):
    #LOAD DEMONSTRATIONS INTO BUFFER
    env = env()

    def load(value:int):
        if value == 0:
            return

        Reader = Demos(env=env, env_name=env_name, filename=read_demo_filename, agent = "DQN", number_of_demonstrations = 60, seed=10)
        demo_dict = Reader.readDemos(read_demo_filename)

        if value == 0:
            return
        
        for demo_idx in range(value):
            done=False
            print("LOADING:", demo_idx)

            for step in range(demo_dict[demo_idx]["steps"]-1):
                reward = demo_dict[demo_idx]["rewards"][step]
                action = demo_dict[demo_idx]["actions"][step]
                state = demo_dict[demo_idx]["states"][step]

                try:
                    next_state = demo_dict[demo_idx]["states"][step+1]
                except:
                    next_state=None
                    done=True
                
                agent.remember(state, action, reward, next_state, done, priority=200)

    score_hist = []
    eval_hist = []
    epsilon = eps_start

    load(demo_load)

    bar_format = '{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]'
    # bar_format = '{l_bar}{bar:10}{r_bar}'
    pbar = trange(n_episodes, unit="ep", bar_format=bar_format, ascii=True)
    past_eval = 0.0
    for idx_epi in trange(n_episodes):
        state = env.reset()[0]
        # print(state.shape)
        score = 0
        for idx_step in range(max_steps):
            action = agent.chooseAction(state, epsilon)
            next_state, reward, done, _ , _= env.step(action)
            agent.remember(state, action, reward, next_state, done, priority = None)
            state = next_state


            # TODO Fix the rewards if it hits the max steps
            # if idx_step == max_steps - 1:
            #     reward = -50
            #     break

            if done:
                break

            score += reward


        score_hist.append(score)
        score_avg = np.mean(score_hist[-50:])
        eval_avg = np.mean(eval_hist[-50:])

        epsilon = max(eps_end, epsilon * eps_decay)

        pbar.set_postfix_str(f"Score: {score: 7.2f}, Epsilon: {epsilon}, 10 eval avg: {eval_avg: 7.2f}, 10 score avg: {score_avg: 7.2f}")
        pbar.update(1)

        if idx_epi%20 == 0:
            eval = evaluatePolicy(env=env, agent=agent, loop=1)
            eval_hist.append(eval)
            if eval > past_eval:
                torch.save({
                    'epoch': idx_epi,
                    'model_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
            
            }, write_policy_filename + "_" + str(int(eval*100)))
                past_eval = eval
                
        if len(score_hist) >= 100:
            if score_avg >= target:
                break


    if (idx_epi+1) < n_episodes:
        print("\nTarget Reached!")
    else:
        print("\nDone!")
    

    return score_hist, eval_hist



# <--------------------------------------------------------------Running the training---------------------------------------------------------------------------->

def evaluate_models():

    POLICY_PATH = read_policy_filename

    checkpoint_p = torch.load(POLICY_PATH)

    agent = create_agent()

    agent.policy_net.load_state_dict(checkpoint_p['model_state_dict'])

    agent.optimizer.load_state_dict(checkpoint_p['optimizer_state_dict'])

    # policy = policy.to(device)
    # now individually transfer the optimizer parts...
    for state in agent.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return agent
    

def create_agent():
    agent = DQN(
        n_observations = n_observations,
        n_actions = n_actions,
        batch_size = BATCH_SIZE,
        lr = LR,
        gamma = GAMMA,
        mem_size = MEMORY_SIZE,
        learn_step = LEARN_STEP,
        tau = TAU,
        )
    return agent

if str(device) == "cuda":
    torch.cuda.empty_cache()

def TextOnImg(img, score):
    img = Image.fromarray(img)
    # font = ImageFont.truetype('Arial', 18)
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), f"Score={score: .2f}", fill=(255, 255, 255))

    return np.array(img)


def save_frames_as_gif(frames, filename, path="gifs/"):
    if not os.path.exists(path):
        os.makedirs(path)
        
    print("Saving gif...", end="")
    imageio.mimsave(path + filename + ".gif", frames, fps=30)

    print("Done!")

def gym2gif(env, agent, filename="animated_gif", loop=3):
    frames = []
    env= env()
    for i in range(loop):
        state  = env.reset()
        score = 0
        for idx_step in range(1000):
            frame = env.render()
            frames.append(TextOnImg(frame, score))
            action = agent.chooseAction(state, epsilon=0)
            state, reward, done, info = env.step(action)
            score += reward
            if done:
                break
    env.close()
    save_frames_as_gif(frames, filename=filename)


def collectOracle():
    agent = evaluate_models()
    #TODO THE 'env' may throw an error
    Collector = Demos(env=env, env_name=env_name, filename=write_oracle_filename, agent = agent, number_of_demonstrations = 1, seed=10)
    evaluation = Collector.collectAgentDemos(num_demos=1, agent=agent)
    print("Evaluation Accuracy:", evaluation)

def collectGif():
    agent = evaluate_models()

    gym2gif(env, agent, loop=3, filename=read_oracle_filename)

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)


def collect_demonstrations(num_demos=30, experiment=False):
    agent = create_agent()

    print("Collecting and Writing To: {}".format(write_demo_filename))


    Collector = Demos(env=env, env_name=env_name, filename=write_demo_filename, agent = agent, number_of_demonstrations = 30, seed=10, experiment=experiment)

    Collector.collectDemos()

def render_demonstrations(num_demos=30, experiment=False):
    agent = create_agent()

    print("Rendering the Demonstration Titled: {}".format(read_demo_filename))

    Renderer = Demos(env=env, env_name=env_name, filename=read_demo_filename, agent = agent, number_of_demonstrations = num_demos, seed=10, experiment=experiment)

    Renderer.renderDemos(demo_file_name=read_demo_filename)

def evaluate_demonstrations(num_demos=30, experiment=False):
    print("evaluating")
    agent = create_agent()

    print("Rendering the Demonstration Titled: {}".format(read_demo_filename))

    Renderer = Demos(env=env, env_name=env_name, filename=read_demo_filename, agent = agent, number_of_demonstrations = num_demos, seed=10, experiment=experiment)

    Renderer.evalDemos(demo_file_name=read_demo_filename)



import csv


def main_train_trials(oracle=False, gif=False):
    for trial, demo in enumerate(num_demo_trials):
        print("Starting with {} Demonstrations".format(demo))

        for experiment in range(experiment_trials):

            print("Experiment {}".format(experiment))

            agent = create_agent()

            score_hist, eval_hist = train(env, agent, n_episodes=EPISODES, target=TARGET_SCORE, chkpt=SAVE_CHKPT, demo_load=demo) 
            
            if not os.path.exists("data/results/{}/rewards".format(experiment_name)):
                os.makedirs("data/results/{}/rewards".format(experiment_name))
                os.makedirs("data/results/{}/eval".format(experiment_name))

            with open("data/results/{}/rewards/D{}_T{}_E{}.csv".format(experiment_name, demo, trial, experiment), 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(score_hist)

            with open("data/results/{}/eval/D{}_T{}_E{}.csv".format(experiment_name, demo, trial, experiment), 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(eval_hist)

    if oracle:
        collectOracle()
    if gif:
        collectGif()


def watch_experiment():
    render_demonstrations(experiment=True)

def play_experiment():
    collect_demonstrations(num_demos=30, experiment=True)

#EVALUATE POLICY
def evaluate_policy(loop=1):
    agent = evaluate_models()
    env = FlappyBirdEnv(render_mode="human")
    wins=0
    for i in range(loop):
        state  = env.reset()
        for idx_step in range(500):
            action = agent.chooseAction(state, epsilon=0)
            state, reward, done , info = env.step(action)
            if done:
                if win:
                    wins += 1
                break
    env.close()

    return wins/loop


main_train_trials(oracle=False, gif=False)


# collectOracle()
# watch_experiment()
# play_experiment()
# evaluate_demonstrations()
# evaluate_policy()

