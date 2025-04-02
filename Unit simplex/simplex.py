import numpy as np
import matplotlib.pyplot as plt
import random as random

def replicator_dynamics_2x2(A, B, resolution=21):
    """
    Affiche le champ de vecteurs de la dynamique réplicative pour un jeu 2x2.
    """
    # Création d'une grille 2D pour x et y
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calcul des variations pour chaque point de la grille
    x_point = X * (1 - X) * (Y * (A[0, 0] - A[1, 0]) + (1 - Y) * (A[0, 1] - A[1, 1]))
    y_point = Y * (1 - Y) * (X * (B[0, 0] - B[0, 1]) + (1 - X) * (B[1, 0] - B[1, 1]))
    
    # Création du champ de vecteurs
    plt.quiver(X, Y, x_point, y_point, color='black', angles='xy')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('p (Population ligne joue stratégie 0)')
    plt.ylabel('q (Population colonne joue stratégie 0)')
    plt.gca().set_aspect('equal', adjustable='box')

# --- Définition de 3 jeux classiques ---

# 1) Prisoner's Dilemma (payoff row, payoff column)
A_PD = np.array([[3, 0],
                 [5, 1]])
B_PD = np.array([[3, 5],
                 [0, 1]])

# 2) Stag Hunt
A_SH = np.array([[4, 0],
                 [3, 3]])
B_SH = np.array([[4, 3],
                 [0, 3]])

# 3) Matching Pennies
A_MP = np.array([[1, -1],
                 [-1, 1]])
B_MP = -A_MP  # Jeu à somme nulle

# 4) Autre
A_A = np.array([[6, 0],
                 [0, 3]])
B_A = np.array([[6, 15],
                 [0, 1]])

# --- Affichage de chaque jeu sur une figure distincte ---

# Prisoner's Dilemma
plt.figure(figsize=(6, 6))
replicator_dynamics_2x2(A_PD, B_PD, resolution=21)
plt.title("Prisoner's Dilemma")
plt.show()

# Stag Hunt
plt.figure(figsize=(6, 6))
replicator_dynamics_2x2(A_SH, B_SH, resolution=21)
plt.title("Stag Hunt")
plt.show()

# Matching Pennies
plt.figure(figsize=(6, 6))
replicator_dynamics_2x2(A_MP, B_MP, resolution=21)
plt.title("Matching Pennies")
plt.show()


#####################################################################################################


class Agent:
    def __init__(self,nom, m, epsilon, decay, algo):

        self.nom = nom
        self.m = m
        self.epsilon = epsilon
        self.decay = decay
        self.algo = algo

        self.nbs_A = 0
        self.nbs_B = 0

        self.somme_A = 0
        self.somme_B = 0

        self.moyenne_A = 0
        self.moyenne_B = 0

        self.histo_actions = []
        self.histo_recompenses = []
        self.histo_probabilities = []

        self.id = -1
        self.action = ""

    def jouer_action_epsilon(self):

        # À modifier le choix initial probablement
        if self.nbs_A <= self.m and self.nbs_B <= self.m:
            self.action = random.sample(['A','B'],1)[0]
            self.histo_probabilities.append(0.5)
        elif self.nbs_A <= self.m:
            self.action = "A"
            self.histo_probabilities.append(1)
        elif self.nbs_B <= self.m:
            self.action = "B"
            self.histo_probabilities.append(0)

        else:
            e = self.epsilon
            if self.decay == True:
                e = min(self.epsilon,0.9999**(self.nbs_A+self.nbs_B))
            if e < random.uniform(0,1):
                self.moyenne_A = self.somme_A/self.nbs_A
                self.moyenne_B = self.somme_B/self.nbs_B

                moyenne_recompense = [self.moyenne_A,self.moyenne_B]
                maximum = moyenne_recompense.index(max(moyenne_recompense))
                if maximum == 0:
                    self.action = "A"
                    self.histo_probabilities.append(1-e/2)
                elif maximum == 1:
                    self.action = "B"
                    self.histo_probabilities.append(e/2)
            else :
                self.action = random.sample(['A','B'],1)[0]
                self.histo_probabilities.append(0.5)


    def jouer_action_exp3(self):
        self.gamma = self.epsilon

        if self.nbs_A == 0 and self.nbs_B == 0 :
            self.w_A = 1
            self.w_B = 1
        else:
            self.gamma = min(1,np.sqrt(np.log(2)/2/(self.nbs_A+self.nbs_B)))

        somme = self.w_A + self.w_B
        self.p_A = (1-self.gamma)*self.w_A/somme + self.gamma/2
        self.p_B = (1-self.gamma)*self.w_B/somme + self.gamma/2
        self.p = (self.p_A,self.p_B)
        self.action = random.choices(['A','B'], weights=(self.p),k=1)[0]
        self.histo_probabilities.append(float(self.p_A))

    def update_action(self):
        self.histo_actions.append(self.action)
        if self.action == 'A':
            self.nbs_A += 1
        elif self.action == 'B':
            self.nbs_B += 1
        if self.nom == "Agent_1":
            print(self.nbs_A,self.nbs_B)


    def jouer_action(self):
        if self.algo == "epsilon":
            self.jouer_action_epsilon()

        elif self.algo == "exp3":
            self.jouer_action_exp3()

        else:
            raise Exception("Not a valid name for algo")
        self.update_action()


    def afficher_histo_actions(self):
        print(self.histo_actions)
        print(self.histo_recompenses)

    def afficher_id(self):
        print(self.id)
        print(self.nbs_A)
        print(self.nbs_B)

    def update_recompense(self,recompense):
        self.histo_recompenses.append(recompense)
        if self.action == "A":
            self.somme_A += recompense
        if self.action == "B":
            self.somme_B += recompense

        if self.algo == "exp3":
            if self.action == "A":
                self.w_A *= np.exp(self.gamma*recompense/self.p_A/2)
            if self.action == "B":
                self.w_B *= np.exp(self.gamma*recompense/self.p_B/2)

            somme2 = self.w_A + self.w_B 
            self.w_A = self.w_A/somme2
            self.w_B = self.w_B/somme2


class Environnement:
    def __init__(self,AA,BB):
        self.A = AA
        self.B = BB
        self.agents = []
        self.agents_count = 0
        self.agents_step = []

    def ajouter_agents(self,agent):
        self.agents_count += 1
        self.agents.append(agent)
        agent.id = self.agents_count

    def step(self):
        for i in range(0,  self.agents_count - 1, 2):
            self.agents[i].jouer_action()
            self.agents[i+1].jouer_action()
            action_agent1 = self.agents[i].action
            action_agent2 = self.agents[i+1].action

            
            if (action_agent1, action_agent2) == ('A','A'):
                self.agents[i].update_recompense(self.A[0,0])
                self.agents[i+1].update_recompense(self.B[0,0])

            if (action_agent1, action_agent2) == ('A','B'):
                self.agents[i].update_recompense(self.A[0,1])
                self.agents[i+1].update_recompense(self.B[0,1])

            if (action_agent1, action_agent2) == ('B','A'):
                self.agents[i].update_recompense(self.A[1,0])
                self.agents[i+1].update_recompense(self.B[1,0])

            if (action_agent1, action_agent2) == ('B','B'):
                self.agents[i].update_recompense(self.A[1,1])
                self.agents[i+1].update_recompense(self.B[1,1])

if __name__ == "__main__":
    A = A_A
    B = B_A

    random.seed(5)

    for i in range(10):
        # L'environnement
        environnement = Environnement(A, B)

        # Ajout des agents dans l'environnement
        agent1 = Agent("Agent_1", m=1, epsilon=0.03, decay=True, algo="exp3")
        environnement.ajouter_agents(agent1)
        agent2 = Agent("Agent_2", m=1, epsilon=0.03, decay=True, algo="exp3")
        environnement.ajouter_agents(agent2)

        # Exécution des expériences
        nombre_essai = 300
        for i in range(nombre_essai):
            environnement.step()

        # Affichage des historiques de probabilités pour vérification
        for agent in environnement.agents:
            print(f"{agent.nom} - Historique des probabilités:")
            print(agent.histo_probabilities)

        # Extraction des probabilités d'opter pour "A" (premier élément du tuple) pour chaque agent
        x_line = [p for p in agent1.histo_probabilities]
        y_line = [p for p in agent2.histo_probabilities]

        # Matching Pennies
        plt.figure(figsize=(6, 6))
        replicator_dynamics_2x2(A, B, resolution=21)
        plt.title("Matching Pennies")

        # Tracé de la ligne reliant les points (probabilité de choisir A pour Agent_1 vs Agent_2)
        plt.plot(x_line, y_line, marker='o', linestyle='-', color='red')
        plt.xlabel("Probabilité de choisir A (Agent 1)")
        plt.ylabel("Probabilité de choisir A (Agent 2)")
        plt.show()





"""
    # Avec moyenne

    A = A_MP
    B = B_MP

    num_trials = 10
    num_steps = 300

    # Pour stocker, pour chaque essai, l'historique (dans le temps) des probabilités de choisir A pour chaque agent.
    all_prob_A_agent1 = []
    all_prob_A_agent2 = []

    for trial in range(num_trials):

        # Initialisation de l'environnement et des agents avec le jeu Matching Pennies
        env = Environnement(A_MP, B_MP)
        agent1 = Agent("Agent_1", m=2, epsilon=0.03, decay=True, algo="exp3")
        agent2 = Agent("Agent_2", m=2, epsilon=0.03, decay=True, algo="exp3")
        env.ajouter_agents(agent1)
        env.ajouter_agents(agent2)

        # Exécution de l'expérience
        for _ in range(num_steps):
            env.step()

        # Extraction de l'historique des probabilités pour l'action A (première composante)
        prob_A_agent1 = [p for p in agent1.histo_probabilities]
        prob_A_agent2 = [p for p in agent2.histo_probabilities]
        all_prob_A_agent1.append(prob_A_agent1)
        all_prob_A_agent2.append(prob_A_agent2)

    # Conversion en tableaux numpy pour faciliter la moyenne
    all_prob_A_agent1 = np.array(all_prob_A_agent1)  # Dimension: (num_trials, num_steps)
    all_prob_A_agent2 = np.array(all_prob_A_agent2)

    # Calcul de la moyenne (pour chaque pas de temps) sur les essais
    avg_prob_A_agent1 = np.mean(all_prob_A_agent1, axis=0)
    avg_prob_A_agent2 = np.mean(all_prob_A_agent2, axis=0)

    # --- Tracé du résultat ---
    plt.figure(figsize=(10, 5))
    plt.plot(avg_prob_A_agent1, label="Agent_1 - moyenne P(A)")
    plt.plot(avg_prob_A_agent2, label="Agent_2 - moyenne P(A)")
    plt.xlabel("Pas de temps")
    plt.ylabel("Probabilité moyenne de choisir A")
    plt.title("Moyenne des probabilités (P(A)) sur 10 essais")
    plt.legend()
    plt.show()

    # Matching Pennies
    plt.figure(figsize=(6, 6))
    replicator_dynamics_2x2(A_MP, B_MP, resolution=21)
    plt.title("Matching Pennies")


    # Liste de points à relier par une ligne
    x_line = avg_prob_A_agent1
    y_line = avg_prob_A_agent2
    # Ajout de la ligne dans le graphique
    plt.plot(x_line, y_line, marker='o', linestyle='-', color='red')

    plt.show()"
"""