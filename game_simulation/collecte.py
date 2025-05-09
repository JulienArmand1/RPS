import matplotlib.pyplot as plt
import numpy as np
from .environnement import Environnement
from .agent import Agent
from .visualisation import replicator_dynamics_2x2
from .games import A_MP, B_MP, A_PD, B_PD, A_A, B_A, A_A2, B_A2

def collecte(A, B, titre, algo1, algo2):

    histo_agent1 = []
    histo_agent2 = []
    histo_r_agent1 = []
    histo_r_agent2 = []

    for _ in range(200):
        env = Environnement(A, B)
        a1 = Agent("Agent_1", m=1, epsilon_init=0.2, decay=True, algo=algo1)
        a2 = Agent("Agent_2", m=1, epsilon_init=0.2, decay=True, algo=algo2)
        env.ajouter_agents(a1)
        env.ajouter_agents(a2)

        for _ in range(300):
            env.step()

        histo_agent1.append(a1.hist_actions)
        histo_agent2.append(a2.hist_actions)
        histo_r_agent1.append(a1.hist_rewards)
        histo_r_agent2.append(a2.hist_rewards)


    return histo_agent1, histo_agent2, histo_r_agent1, histo_r_agent2


def main():
    histo_agent1, histo_agent2, histo_r_agent1, histo_r_agent2 = collecte(A_MP, B_MP, "Prisonner's dilema exp3", "exp3", "epsilon")
    histo_PD_exp3_vs_epsilon = np.array(histo_agent1)
    histo_PD_exp3_vs_epsilon_reward = np.array(histo_r_agent1)
    print(np.mean(histo_PD_exp3_vs_epsilon_reward))
    print(np.std(np.mean(histo_PD_exp3_vs_epsilon_reward, axis=1)))
    with open('histo_PD_exp3_vs_epsilon.npy', 'wb') as f:
        np.save(f, histo_PD_exp3_vs_epsilon)

    histo_agent1, histo_agent2, histo_r_agent1, histo_r_agent2  = collecte(A_MP, B_MP, "Prisonner's dilema exp3", "exp3", "exp3")
    histo_PD_exp3_vs_exp3 = np.array(histo_agent1)
    histo_PD_exp3_vs_exp3_reward = np.array(histo_r_agent1)
    print(np.mean(histo_PD_exp3_vs_exp3_reward))
    print(np.std(np.mean(histo_PD_exp3_vs_exp3_reward, axis=1)))
    histo_PD_exp3_vs_exp3 = np.array(histo_agent1)
    with open('histo_PD_exp3_vs_exp3.npy', 'wb') as f:
        np.save(f, histo_PD_exp3_vs_exp3)

if __name__ == "__main__":
    main()
