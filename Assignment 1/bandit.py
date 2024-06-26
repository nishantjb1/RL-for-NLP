import numpy as np
import matplotlib.pyplot as plt

def button1():
    return np.random.normal(2, 1)

def button2():
    return 5 if np.random.rand() < 0.5 else -6

def button3():
    return np.random.poisson(2)

def button4():
    return np.random.exponential(3)

def button5():
    return np.random.choice([button1(), button2(), button3(), button4()])

buttons = [button1, button2, button3, button4, button5]

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)  
        self.values = np.zeros(n_arms)  

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value

def simulate_bandit(epsilon, n_episodes, n_steps):
    agent = EpsilonGreedy(n_arms=len(buttons), epsilon=epsilon)
    total_rewards = []

    for episode in range(n_episodes):
        total_reward = 0

        for step in range(n_steps):
            chosen_arm = agent.select_arm()
            reward = buttons[chosen_arm]()
            agent.update(chosen_arm, reward)
            total_reward += reward

        total_rewards.append(total_reward)
    
    return total_rewards


n_runs = 25
n_episodes = 1000
n_steps = 100

epsilons = [0, 0.1, 0.01]
results = {}

for epsilon in epsilons:
    avg_rewards = np.zeros(n_episodes)
    
    for run in range(n_runs):
        
        rewards = simulate_bandit(epsilon, n_episodes, n_steps)
        avg_rewards += np.array(rewards)
    
    avg_rewards /= n_runs
    results[epsilon] = avg_rewards

plt.figure(figsize=(10, 6))

for epsilon, rewards in results.items():
    plt.plot(rewards, label=f'ε = {epsilon}')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()