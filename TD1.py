import numpy as np


int_to_char = {
    0: 'u',  
    1: 'r',  
    2: 'd',  
    3: 'l'   
}

actions = {'UP': -4, 'RIGHT': 1, 'DOWN': 4, 'LEFT': -1}
policy_one_step_look_ahead = {
    0: [-1, 0],
    1: [0, 1],
    2: [1, 0],
    3: [0, -1]
}

def policy_int_to_char(pi, n):
    pi_char = [[''] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0 or i == n - 1 and j == n - 1:
                continue
            pi_char[i][j] = int_to_char[pi[i, j]]
    return np.array(pi_char)

def policy_evaluation(n, pi, v, Gamma, threshold):
    while True:
        delta = 0
        for i in range(n):
            for j in range(n):
                if (i == 0 and j == 0) or (i == n - 1 and j == n - 1):
                    continue
                old_value = v[i, j]
                action = pi[i, j]
                ni, nj = i + policy_one_step_look_ahead[action][0], j + policy_one_step_look_ahead[action][1]
                if ni < 0 or ni >= n or nj < 0 or nj >= n:
                    ni, nj = i, j
                v[i, j] = -1 + Gamma * v[ni, nj]
                delta = max(delta, abs(old_value - v[i, j]))
        if delta < threshold:
            break
    return v


def policy_improvement(n, pi, v, Gamma):
    policy_stable = True
    for i in range(n):
        for j in range(n):
            if (i == 0 and j == 0) or (i == n - 1 and j == n - 1):
                continue
            old_action = pi[i, j]
            action_values = []
            for action in range(4):  # 4 actions
                ni, nj = i + policy_one_step_look_ahead[action][0], j + policy_one_step_look_ahead[action][1]
                if ni < 0 or ni >= n or nj < 0 or nj >= n:
                    ni, nj = i, j
                action_values.append(-1 + Gamma * v[ni, nj])
            best_action = np.argmax(action_values)
            pi[i, j] = best_action
            if old_action != best_action:
                policy_stable = False
    return pi, policy_stable

def policy_initialization(n):
    pi = np.random.randint(0, 4, size=(n, n))
    pi[0, 0] = -1  
    pi[n - 1, n - 1] = -1  
    return pi


def policy_iteration(n, Gamma, threshold):
    pi = policy_initialization(n)
    v = np.zeros((n, n))
    while True:
        v = policy_evaluation(n, pi, v, Gamma, threshold)
        pi, policy_stable = policy_improvement(n, pi, v, Gamma)
        if policy_stable:
            break
    return pi, v


n = 4
Gamma_values = [0.8, 0.9, 1]
threshold = 1e-4

for Gamma in Gamma_values:
    pi, v = policy_iteration(n, Gamma, threshold)
    pi_char = policy_int_to_char(pi, n)
    
    print(f"\nGamma = {Gamma}\n")
    print("Politique optimale :\n", pi_char)
    print("\nValeur optimale :\n", v)
