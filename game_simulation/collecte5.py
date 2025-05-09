import matplotlib.pyplot as plt
import numpy as np
from .environnement import Environnement
from .agent import Agent
from .visualisation import replicator_dynamics_2x2
from .games import A_MP, B_MP, A_PD, B_PD, A_A, B_A, A_A2, B_A2

def collecte(A, B, algo1, algo2, n_runs=5000, n_steps=300):
    histo_agent1 = []
    histo_agent2 = []
    histo_r_agent1 = []
    histo_r_agent2 = []
    histo_p_agent1 = []
    histo_p_agent2 = []
    o=0
    while o <= n_runs:
        env = Environnement(A, B)
        a1 = Agent("Agent_1", m=1, epsilon_init=0.1, decay=True, algo=algo1)
        a2 = Agent("Agent_2", m=1, epsilon_init=0.1, decay=True, algo=algo2)
        env.ajouter_agents(a1)
        env.ajouter_agents(a2)

        for _ in range(n_steps):
            env.step()
        if sum(a1.hist_rewards)==0:
            o+=1
            histo_agent1.append(a1.hist_actions)
            histo_agent2.append(a2.hist_actions)
            histo_r_agent1.append(a1.hist_rewards)
            histo_r_agent2.append(a2.hist_rewards)
            histo_p_agent1.append(a1.hist_probas)
            histo_p_agent2.append(a2.hist_probas)

    return np.array(histo_agent1), np.array(histo_agent2), np.array(histo_r_agent1), np.array(histo_r_agent2),np.array(histo_p_agent1), np.array(histo_p_agent2)

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


def plot_action_proportions_v2(histo_proba, title):
    """
    histo_proba : np.array de shape (n_runs, n_steps), valeurs 1 pour A et 0 pour B
    title       : titre du graphique
    """
    # Calcul des proportions et des écarts-types
    prop_A = np.mean(histo_proba, axis=0)
    std_A  = np.std(histo_proba, axis=0)
    prop_B = 1.0 - prop_A
    std_B  = std_A  # même écart-type

    x = np.arange(histo_proba.shape[1])

    plt.figure(figsize=(8, 4))
    # Tracé des intervalles ±1 std
    plt.fill_between(x, prop_A - std_A, prop_A + std_A,
                     alpha=0.2, label='±1 std A')
    plt.fill_between(x, prop_B - std_B, prop_B + std_B,
                     alpha=0.2, label='±1 std B')
    # Tracé des moyennes
    plt.plot(x, prop_A, label='Action A')
    plt.plot(x, prop_B, label='Action B')

    plt.xlabel('Étape de simulation')
    plt.ylabel('Proportion')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_action_proportions_v3(histo_proba, title, n_runs=10):
    """
    histo_proba : np.array de shape (n_runs_tot, n_steps), valeurs 1 pour A et 0 pour B
    title       : titre du graphique
    n_runs      : nombre de runs aléatoires à tracer (par défaut 10)
    """
    n_runs = min(n_runs, histo_proba.shape[0])
    # Sélection aléatoire de n_runs indices sans remplacement
    indices = np.random.choice(histo_proba.shape[0], size=n_runs, replace=False)
    selected = histo_proba[indices]  # shape (n_runs, n_steps)

    x = np.arange(histo_proba.shape[1])

    plt.figure(figsize=(8, 4))
    for idx, run in zip(indices, selected):
        plt.plot(x, run, alpha=0.7, label=f'Run {idx}')
    plt.xlabel('Étape de simulation')
    plt.ylabel('Action (1 = A, 0 = B)')
    plt.yticks([0, 1], ['B', 'A'])
    plt.title(title)
    plt.legend(loc='upper right', ncol=2, fontsize='small')
    plt.tight_layout()
    plt.show()

def run_experiment(algo1: str, algo2: str, tag: str, title: str,
                   n_runs: int = 5000, n_steps: int = 300):
    """Exécute l'expérience, affiche stats, sauve et trace."""
    # 1) Collecte
    histo1, histo2, r1, r2, p1, p2 = collecte(A_MP, B_MP, algo1, algo2, n_runs, n_steps)

    # 2) Stats de la récompense par run pour Agent_1
    means = np.mean(r1, axis=1)                  # (n_runs,)
    mean_reward = means.mean()
    std_reward  = means.std(ddof=1)              # écart-type échantillon
    sem_reward  = std_reward / np.sqrt(n_runs)   # SEM

    # 3) Affichage
    print(f"[{tag}] {algo1} vs {algo2}")
    print(f"  • Moyenne récompense Agent 1 : {mean_reward:.4f}")
    print(f"  • Écart-type moyen         : {std_reward:.4f}")
    print(f"  • Erreur-type (SEM)        : {sem_reward:.4f}")

    # 4) Sauvegarde
    np.save(f'histo_PD_{tag}_v2.npy', histo1)

    # 5) Plot
    plot_action_proportions(histo1, title)
    plot_action_proportions_v2(p1, title)
    plot_action_proportions_v3(p1, title)

def main():
    experiments = [
        ("exp3",    "epsilon", "exp3_vs_epsilon", "Agent 1 actions (exp3 vs ε-greedy)"),
        ("exp3",    "exp3",    "exp3_vs_exp3",    "Agent 1 actions (exp3 vs exp3)"),
        ("exp3",    "ucb",     "exp3_vs_ucb",     "Agent 1 actions (exp3 vs UCB)"),
        ("exp3",    "ftl",     "exp3_vs_ftl",     "Agent 1 actions (exp3 vs FTL)"),
    ]

    for algo1, algo2, tag, title in experiments:
        run_experiment(algo1, algo2, tag, title)

if __name__ == "__main__":
    main()
