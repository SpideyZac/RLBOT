from tools import  *
from objects import *
from routines import *
import numpy as np
import random


#This file is for strategy

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class nerual_network:
    def __init__(self, numofinputs, numofhidden, numofoutputs):
        self.hiddenweights = []
        self.outputweights = []

        for i in range(0, numofhidden):
            self.hiddenweights.append([])
            for _ in range(0, numofinputs):
                self.hiddenweights[i].append(random.randint(-5, 5))

        for i in range(0, numofoutputs):
            self.outputweights.append([])
            for _ in range(0, numofhidden):
                self.outputweights[i].append(random.randint(-5, 5))

    def get_output(self, inputs):
        self.hiddenoutputs = []
        for i in range(0, len(self.hiddenweights)):
            self.hiddenoutputs.append(sigmoid(np.dot(self.hiddenweights[i], inputs)))

        self.outputoutputs = []
        for i in range(0, len(self.outputweights)):
            self.outputoutputs.append(sigmoid(np.dot(self.hiddenoutputs, self.outputweights[i])))
        
        return self.outputoutputs

def crossover(PA, PB):
    child = nerual_network(6, 4, 3)
    ndc = random.randint(0, len(PA.hiddenweights) - 1)
    noc = random.randint(0, len(PA.outputweights) - 1)

    for i in range(0, len(PA.hiddenweights)):
        if i < ndc:
            child.hiddenweights[i] = PA.hiddenweights[i]
        else:
            child.hiddenweights[i] = PB.hiddenweights[i]

    for i in range(0, len(PA.outputweights)):
        if i < noc:
            child.outputweights[i] = PA.outputweights[i]
        else:
            child.outputweights[i] = PB.outputweights[i]

    return child

def mutate(INDIVIDUAL):
    for i in range(0, len(INDIVIDUAL.hiddenweights)):
        for j in range(0, len(INDIVIDUAL.hiddenweights[i])):
            INDIVIDUAL.hiddenweights[i][j] += random.randint(-5, 5)

    for i in range(0, len(INDIVIDUAL.outputweights)):
        for j in range(0, len(INDIVIDUAL.outputweights[i])):
            INDIVIDUAL.outputweights[i][j] += random.randint(-5, 5)

    return INDIVIDUAL

networks = [nerual_network(6, 4, 3), nerual_network(6, 4, 3), nerual_network(6, 4, 3), nerual_network(6, 4, 3), nerual_network(6, 4, 3)]
fitness = [1, 1, 1, 1, 1]
'''
inputs:
oponentdistancetoball
distancetoball
balldistancetogoal
balldistancetofoegoal
boost
distanceToBoost

outpust:
goToBall
goBackToGoal
getBoost
'''

curentNetwork = -1

myTeamScore = 0
pastTeamScore = 0
opoTeamScore = 0
pastOpoTeamScore = 0

class Bob(GoslingAgent):
    def run(agent):
        if len(agent.stack) < 1:
            global pastTeamScore
            global myTeamScore
            global curentNetwork
            global networks
            global fitness
            global opoTeamScore
            global pastOpoTeamScore
            myTeamScore = agent.game.friend_score
            opoTeamScore = agent.game.foe_score
            if myTeamScore > pastTeamScore:
                fitness[curentNetwork] += 10
                pastTeamScore = myTeamScore
            if opoTeamScore > pastOpoTeamScore:
                fitness[curentNetwork] -= 10
            if opoTeamScore > pastOpoTeamScore and agent.ball.latest_touched_team == agent.team:
                fitness[curentNetwork] -= 50
            pastOpoTeamScore = opoTeamScore
            if curentNetwork + 1 < 5:
                curentNetwork += 1
            else:
                selection = []
                for i in range(0, len(fitness)):
                    for _ in range(0, fitness[i]):
                        selection.append(networks[i])
                for i in range(0, len(networks)):
                    networks[i] = mutate(crossover(random.choice(selection), random.choice(selection)))
                curentNetwork = 0

            opponentToBall = (agent.ball.location - agent.foes[0].location).magnitude()
            meToBall = (agent.ball.location - agent.me.location).magnitude()
            ballToGoal = (agent.friend_goal.location - agent.ball.location).magnitude()
            ballToFoeGoal = (agent.foe_goal.location - agent.ball.location).magnitude()
            bestDistance = 100

            nnOutputs = networks[curentNetwork].get_output([opponentToBall, meToBall, ballToGoal, ballToFoeGoal, agent.me.boost, bestDistance])
            bestScore = float("-inf")
            bestIndex = None
            for i in range(0, len(nnOutputs)):
                if nnOutputs[i] > bestScore:
                    bestScore = nnOutputs[i]
                    bestIndex = i

            if agent.kickoff_flag:
                agent.push(kickoff())
                
            elif bestIndex == 0:
                targets = {"goal": (agent.foe_goal.left_post, agent.foe_goal.right_post)}
                shots = find_hits(agent, targets)

                if len(shots["goal"]) > 0:
                    agent.push(shots["goal"][0])
                else:
                    defaultPD(agent, agent.me.local(agent.ball.location - agent.me.location))
                    defaultThrottle(agent, 3500)

            elif bestIndex == 1:
                if (agent.friend_goal.location - agent.me.location).magnitude() > 100:
                    defaultPD(agent, agent.me.local(agent.friend_goal.location - agent.me.location))
                    defaultThrottle(agent, 3500)

            else:
                bestDistance = float("inf")
                bestIndex = None
                for i in range(0, len(agent.boosts)):
                    if (agent.boosts[i].location - agent.me.location).magnitude() < bestDistance and agent.boosts[i].active == True:
                        bestDistance = (agent.boosts[i].location - agent.me.location).magnitude()
                        bestIndex = i
                if agent.me.boost < 100:
                    agent.push(goto_boost(agent.boosts[bestIndex]))
                else:
                    defaultPD(agent, agent.me.local(agent.ball.location - agent.me.location))
                    defaultThrottle(agent, 3500)