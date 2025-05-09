import matplotlib.pyplot as plt
import numpy as np
from .environnement import Environnement
from .agent import Agent
from .visualisation import replicator_dynamics_2x2
from .games import A_MP, B_MP, A_PD, B_PD, A_A, B_A, A_A2, B_A2

def collecte(A, B, algo1, algo2, n_runs=10000, n_steps=300):
    histo_agent1 = []
    histo_agent2 = []
    histo_r_agent1 = []
    histo_r_agent2 = []

    for _ in range(n_runs):
        env = Environnement(A, B)
        a1 = Agent("Agent_1", m=1, epsilon_init=0.1, decay=True, algo=algo1)
        a2 = Agent("Agent_2", m=1, epsilon_init=0.1, decay=True, algo=algo2)
        env.ajouter_agents(a1)
        env.ajouter_agents(a2)

        for _ in range(n_steps):
            env.step()

        histo_agent1.append(a1.hist_actions)
        histo_agent2.append(a2.hist_actions)
        histo_r_agent1.append(a1.hist_rewards)
        histo_r_agent2.append(a2.hist_rewards)

    return np.array(histo_agent1), np.array(histo_agent2), np.array(histo_r_agent1), np.array(histo_r_agent2)

def plot_action_proportions(histo_actions, title):
    """
    histo_actions : np.array de shape (n_runs, n_steps)
    title         : titre du graphique
    """
    # On suppose que les actions sont stockées sous forme de 'A' et 'B' (chaînes)
    prop_A = np.mean(histo_actions == 0, axis=0)
    prop_B = 1.0 - prop_A

    plt.figure(figsize=(8, 4))
    plt.plot(prop_A, label='Action A')
    plt.plot(prop_B, label='Action B')
    plt.xlabel('Étape de simulation')
    plt.ylabel('Proportion')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # # === Expérience 1 : exp3 vs epsilon ===
    # histo1_exp3_eps, histo2_exp3_eps, r1_exp3_eps, r2_exp3_eps = collecte(
    #     A_MP, B_MP, "exp3", "epsilon"
    # )
    # print(histo1_exp3_eps)
    # print("Moyenne récompense Agent_1 (exp3 vs epsilon) :", np.mean(r1_exp3_eps))
    # print("Écart-type récompense moyenne Agent_1 :", np.std(np.mean(r1_exp3_eps, axis=1)))
    # np.save('histo_PD_exp3_vs_epsilon.npy', histo1_exp3_eps)

    # # Affichage du graphique pour Agent_1
    # plot_action_proportions(
    #     histo1_exp3_eps,
    #     "Proportion des actions pour Agent 1 (exp3 vs ε-greedy)"
    # )

    # # === Expérience 2 : exp3 vs exp3 ===
    # histo1_exp3_exp3, histo2_exp3_exp3, r1_exp3_exp3, r2_exp3_exp3 = collecte(
    #     A_MP, B_MP, "exp3", "exp3"
    # )
    # print("Moyenne récompense Agent_1 (exp3 vs exp3) :", np.mean(r1_exp3_exp3))
    # print("Écart-type récompense moyenne Agent_1 :", np.std(np.mean(r1_exp3_exp3, axis=1)))
    # np.save('histo_PD_exp3_vs_exp3.npy', histo1_exp3_exp3)

    # # Affichage du graphique pour Agent_1
    # plot_action_proportions(
    #     histo1_exp3_exp3,
    #     "Proportion des actions pour Agent 1 (exp3 vs exp3)"
    # )

    # === Expérience 3 : exp3 vs UCB ===
    histo1_exp3_ucb, histo2_exp3_ucb, r1_exp3_ucb, r2_exp3_ucb = collecte(
        A_MP, B_MP, "exp3", "ucb"
    )
    print("Moyenne récompense Agent_1 (exp3 vs ucb) :", np.mean(r1_exp3_ucb))
    print("Écart-type récompense moyenne Agent_1 :", np.std(np.mean(r1_exp3_ucb, axis=1)))
    np.save('histo_PD_exp3_vs_ucb.npy', histo1_exp3_ucb)

    # Affichage du graphique pour Agent_1
    plot_action_proportions(
        histo1_exp3_ucb,
        "Proportion des actions pour Agent 1 (exp3 vs ucb)"
    )

        # === Expérience 4 : exp3 vs ftl ===
    histo1_exp3_ftl, histo2_exp3_ftl, r1_exp3_ftl, r2_exp3_ftl = collecte(
        A_MP, B_MP, "exp3", "ftl"
    )
    print("Moyenne récompense Agent_1 (exp3 vs ftl) :", np.mean(r1_exp3_ftl))
    print("Écart-type récompense moyenne Agent_1 :", np.std(np.mean(r1_exp3_ftl, axis=1)))
    np.save('histo_PD_exp3_vs_ftl.npy', histo1_exp3_ftl)

    # Affichage du graphique pour Agent_1
    plot_action_proportions(
        histo1_exp3_ftl,
        "Proportion des actions pour Agent 1 (exp3 vs ftl)"
    )

if __name__ == "__main__":
    main()
