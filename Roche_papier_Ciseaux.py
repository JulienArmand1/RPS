# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 19:37:47 2025

@author: 14388
"""

import random
import numpy as np
import matplotlib.pyplot as plt



class Agent:
    def __init__(self,nom, m, epsilon, decay, algo):

        self.nom = nom
        self.m = m
        self.epsilon = epsilon
        self.decay = decay
        self.algo = algo

        self.nbs_R = 0
        self.nbs_P = 0
        self.nbs_C = 0

        self.somme_R = 0
        self.somme_P = 0
        self.somme_C = 0

        self.moyenne_R = 0
        self.moyenne_P = 0
        self.moyenne_C = 0

        self.histo_actions = []
        self.histo_recompenses = []
        self.id = -1
        self.action = ""

    def jouer_action_epsilon(self):

        # À modifier le choix initial probablement
        if self.nbs_R <= self.m and self.nbs_C <= self.m and self.nbs_P <= self.m:
            self.action = random.sample(['R','P','C'],1)[0]
        elif self.nbs_R <= self.m:
            self.action = "R"
        elif self.nbs_P <= self.m:
            self.action = "P"
        elif self.nbs_C <= self.m:
            self.action = "C"

        else:
            e = self.epsilon
            if self.decay == True:
                e = min(self.epsilon,0.9999**(self.nbs_R+self.nbs_P+self.nbs_C))
            if e < random.uniform(0,1):
                self.moyenne_R = self.somme_R/self.nbs_R
                self.moyenne_P = self.somme_P/self.nbs_P
                self.moyenne_C = self.somme_C/self.nbs_C
                moyenne_recompense = [self.moyenne_R,self.moyenne_P,self.moyenne_C]
                maximum = moyenne_recompense.index(max(moyenne_recompense))
                if maximum == 0:
                    self.action = "R"
                elif maximum == 1:
                    self.action = "P"
                else:
                    self.action = "C"
            else :
                self.action = random.sample(['R','P','C'],1)[0]


    def jouer_action_exp3(self):
        self.gamma = self.epsilon

        if self.nbs_R == 0 and self.nbs_P == 0 and self.nbs_C == 0:
            self.w_R = 2
            self.w_P = 1
            self.w_C = 1
        else:
            self.gamma = min(1,np.sqrt(np.log(3)/3/(self.nbs_R+self.nbs_P+self.nbs_C)))

        somme = self.w_R + self.w_P +self.w_C
        self.p_R = (1-self.gamma)*self.w_R/somme + self.gamma/3
        self.p_P = (1-self.gamma)*self.w_P/somme + self.gamma/3
        self.p_C = (1-self.gamma)*self.w_C/somme + self.gamma/3
        self.p = (self.p_R,self.p_P,self.p_C)
        self.action = random.choices(['R','P','C'], weights=(self.p),k=1)[0]

    def update_action(self):
        self.histo_actions.append(self.action)
        if self.action == 'R':
            self.nbs_R += 1
        elif self.action == 'P':
            self.nbs_P += 1
        elif self.action == 'C':
            self.nbs_C += 1


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
        print(self.nbs_R)
        print(self.nbs_P)
        print(self.nbs_C)

    def update_recompense(self,recompense):
        self.histo_recompenses.append(recompense)
        if self.action == "R":
            self.somme_R += recompense
        if self.action == "P":
            self.somme_P += recompense
        if self.action == "C":
            self.somme_C += recompense


        if self.algo == "exp3":
            if self.action == "R":
                self.w_R *= np.exp(self.gamma*recompense/self.p_R/3)
            if self.action == "P":
                self.w_P *= np.exp(self.gamma*recompense/self.p_P/3)
            if self.action == "C":
                self.w_C *= np.exp(self.gamma*recompense/self.p_C/3)
            somme2 = self.w_R + self.w_P + self.w_C
            self.w_R = self.w_R/somme2
            self.w_P = self.w_P/somme2
            self.w_C = self.w_C/somme2



class Environnement:
    def __init__(self,recompense):
        self.recompense = recompense
        self.agents = []
        self.agents_count = 0
        self.agents_step = []

    def ajouter_agents(self,agent):
        self.agents_count += 1
        self.agents.append(agent)
        agent.id = self.agents_count

    def step(self):
        random.shuffle(self.agents)
        for i in range(0,  self.agents_count - 1, 2):
            self.agents[i].jouer_action()
            self.agents[i+1].jouer_action()
            action_agent1 = self.agents[i].action
            action_agent2 = self.agents[i+1].action

            conditions_tie = [ ('R', 'R'), ('P', 'P'), ('C', 'C')]
            if (action_agent1, action_agent2) in conditions_tie:
                self.agents[i].update_recompense(self.recompense["tie"])
                self.agents[i+1].update_recompense(self.recompense["tie"])

            conditions_win = [ ('P', 'R'), ('C', 'P'), ('R', 'C')]
            if (action_agent1, action_agent2) in conditions_win:
                self.agents[i].update_recompense(self.recompense["win"])
                self.agents[i+1].update_recompense(self.recompense["lose"])

            conditions_loss = [ ('R', 'P'), ('P', 'C'), ('C', 'R')]
            if (action_agent1, action_agent2) in conditions_loss:
                self.agents[i].update_recompense(self.recompense["lose"])
                self.agents[i+1].update_recompense(self.recompense["win"])


if __name__ == "__main__":


    random.seed(5)

    # L'environnement
    recompense = {
    "lose": 0,
    "tie": 0.5,
    "win": 1
    }
    environnement = Environnement(recompense)

    # Ajout des agents dans l'environnement
    nombre_agents = 96
    for i in range(nombre_agents):
            agent = Agent(f"Agent_{i+1}",m = 10, epsilon = 0.03, decay = False, algo = "exp3")
            environnement.ajouter_agents(agent)

    # Éxécution des expériences
    nombre_essai = 30000
    for i in range(nombre_essai):
        environnement.step()


    # Graphique
    resultat = []
    for i in range(nombre_agents):
        resultat.append([environnement.agents[i].id, environnement.agents[i].histo_actions])

    resultat.sort(key=lambda x: x[0])
    choix_actions = [element[1] for element in resultat]

    choix_actions_array = np.array(choix_actions)
    print(choix_actions_array)

    resu = []
    for i in range(len(choix_actions_array[0,:])):
        R = np.count_nonzero(choix_actions_array[:,i] == 'R')
        P = np.count_nonzero(choix_actions_array[:,i] == 'P')
        C = np.count_nonzero(choix_actions_array[:,i] == 'C')
        resu.append([R,P,C])

    resu2 = np.array(resu)
    R_resu = resu2[:,0]
    P_resu = resu2[:,1]
    C_resu = resu2[:,2]


    plt.plot(R_resu,label="Nbs actions roches")
    plt.plot(P_resu,label="Nbs actions papier")
    plt.plot(C_resu,label="Nbs actions ciseaux")
    plt.legend()
    plt.title('Évolution des actions choisies par n agents en fonction du temps')
    plt.show()


    environnement.step()