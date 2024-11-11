import random
import gym
import numpy as np
import matplotlib.pyplot as plt

def mettre_a_jour_Q(table_Q, etat_actuel, action, recompense, etat_suivant, taux_apprentissage, facteur_discount):
    table_Q[etat_actuel, action] += taux_apprentissage * (recompense + facteur_discount * np.max(table_Q[etat_suivant]) - table_Q[etat_actuel, action])

def choisir_action(table_Q, etat_actuel, taux_exploration):
    return env.action_space.sample() if random.random() < taux_exploration else np.argmax(table_Q[etat_actuel])

env = gym.make("Taxi-v3", render_mode="human")
table_Q = np.zeros([env.observation_space.n, env.action_space.n])

taux_apprentissage, facteur_discount, taux_exploration = 0.1, 0.9, 0.1
nombre_episodes, max_etapes = 500, 100
recompenses_par_episode = []

for episode in range(nombre_episodes):
    etat, _ = env.reset()
    recompense_totale_episode = 0

    for _ in range(max_etapes):
        action = choisir_action(table_Q, etat, taux_exploration)
        etat_suivant, recompense, termine, _, _ = env.step(action)
        mettre_a_jour_Q(table_Q, etat, action, recompense, etat_suivant, taux_apprentissage, facteur_discount)
        etat = etat_suivant
        recompense_totale_episode += recompense
        if termine:
            break

    print(f"Épisode {episode + 1}: Récompense Totale = {recompense_totale_episode}")
    recompenses_par_episode.append(recompense_totale_episode)

print("Récompense Moyenne :", np.mean(recompenses_par_episode))

plt.plot(recompenses_par_episode)
plt.xlabel('Épisodes')
plt.ylabel('Récompense Totale')
plt.title('Récompenses par Épisode')
plt.show()

env.close()