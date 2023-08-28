# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import math

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        INF = float("inf")
        score = 0;
        if action == 'Stop': #dont ever stop exploring
            return -INF

        food_dists = [0]
        #Better if close to the food, reversely correlated with distance to food
        for food in newFood.asList():
            #Euclidean distance
            food_dist = math.sqrt(math.pow((newPos[0] - food[0]), 2) + math.pow((newPos[1] - food[1]), 2))
            food_dists.append(food_dist)

        if len(food_dists)>1:
            score += min(food_dists) * 30 #i picked this as a weight
            score -= len(food_dists)

        #Better if the ghosts are scared more:
        for time in newScaredTimes:
            score += time * 10

        for ghostState in newGhostStates:
            ghost_loc = ghostState.getPosition()
            #print(ghost_loc)
            dist = math.sqrt(math.pow((newPos[0] - ghost_loc[0]), 2) + math.pow((newPos[1] - ghost_loc[1]),2))
            ghost_dists = [0]
            if ghostState.scaredTimer == 0: #strategy to ghost depends on if its scared
                if dist == 0:
                    #definitely avoid meeting a ghost
                    return -INF*10

                if dist > 5: #its ok if its safely far
                    score += 500
            else:
                ghost_dists.append(dist)

        score += min(ghost_dists) * (10)

        #print("score ", score)
        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        return self.bestAction(gameState, 0)

    #this returns the evaluations for the terminal state, min and max
    def value(self, gameState, agentIndex, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose(): #we reached a terminal node
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        if agentIndex >= 1:
            return self.minValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        score = -float("inf")
        next_actions = gameState.getLegalActions(agentIndex)
        for act in next_actions:
            successor_game_state = gameState.generateSuccessor(agentIndex,act) #pacman takes an action
            score = max(score, self.value(successor_game_state, agentIndex+1, depth)) #next turns to ghost
        return score

    def minValue(self,gameState, agentIndex, depth):
        score = float("inf")
        next_actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents()-1: #check if we looked at all ghosts, to increment depth
            depth += 1
            nextAgentIndex = 0 #go back to pacman
        else:
            nextAgentIndex = agentIndex + 1 #next agent is a ghost

        for act in next_actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, act)   #ghost takes an action
            score = min(score, self.value(successor_game_state, nextAgentIndex, depth))  # next turns to ghost
        return score

    def bestAction(self, gameState, agentIndex):
        legalActs = gameState.getLegalActions(agentIndex)
        bestScore = -float('inf')
        best_action = Directions.STOP
        for act in legalActs:
            successor_game_state = gameState.generateSuccessor(agentIndex, act)
            score = self.value(successor_game_state, agentIndex+1, 0)
            if score > bestScore:
                bestScore = score
                best_action = act
        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.bestAction(gameState, 0, -float("inf"), float("inf"))

    #this returns the evaluations for the terminal state, min and max
    def value(self, gameState, agentIndex, depth, alpha, beta):
        if depth == self.depth or gameState.isWin() or gameState.isLose(): #we reached a terminal node
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)
        if agentIndex >= 1:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)

    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        score = -float("inf")
        next_actions = gameState.getLegalActions(agentIndex)
        for act in next_actions:
            successor_game_state = gameState.generateSuccessor(agentIndex,act) #pacman takes an action
            score = max(score, self.value(successor_game_state, agentIndex+1, depth, alpha, beta)) #next turns to ghost
            if score > beta:
                return score
            alpha = max(alpha, score)
        return score

    def minValue(self,gameState, agentIndex, depth, alpha, beta):
        score = float("inf")
        next_actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents()-1: #check if we looked at all ghosts, to increment depth
            depth += 1
            nextAgentIndex = 0 #go back to pacman
        else:
            nextAgentIndex = agentIndex + 1 #next agent is a ghost

        for act in next_actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, act)   #ghost takes an action
            score = min(score, self.value(successor_game_state, nextAgentIndex, depth, alpha, beta))  # next turns to ghost
            if score < alpha:
                return score
            beta = min(beta, score)
        return score

    def bestAction(self, gameState, agentIndex, alpha, beta):
        legalActs = gameState.getLegalActions(agentIndex)
        bestScore = -float('inf')
        best_action = Directions.STOP
        for act in legalActs:
            successor_game_state = gameState.generateSuccessor(agentIndex, act)
            score = self.value(successor_game_state, agentIndex+1, 0, alpha, beta)
            if score > bestScore:
                bestScore = score
                best_action = act
            alpha = max(bestScore, alpha)
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.bestAction(gameState, 0)
#this returns the evaluations for the terminal state, min and max
    def value(self, gameState, agentIndex, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose(): #we reached a terminal node
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        elif agentIndex >= 1:
            return self.minValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        score = -float("inf")
        next_actions = gameState.getLegalActions(agentIndex)
        for act in next_actions:
            successor_game_state = gameState.generateSuccessor(agentIndex,act) #pacman takes an action
            score = max(score, self.value(successor_game_state, agentIndex+1, depth)) #next turns to ghost
        return score

    def minValue(self,gameState, agentIndex, depth):

        score = 0
        next_actions = gameState.getLegalActions(agentIndex)
        p = 1.0/len(next_actions)

        if agentIndex == gameState.getNumAgents()-1: #check if we looked at all ghosts, to increment depth
            depth += 1
            nextAgentIndex = 0 #go back to pacman
        else:
            nextAgentIndex = agentIndex + 1 #next agent is a ghost

        for act in next_actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, act)   #ghost takes an action
            score += self.value(successor_game_state, nextAgentIndex, depth) #* normalize # next turns to ghost

        expect_score = score*p
        return expect_score

    def bestAction(self, gameState, agentIndex):
        legalActs = gameState.getLegalActions(agentIndex)
        bestScore = -float('inf')
        best_action = Directions.STOP
        for act in legalActs:
            successor_game_state = gameState.generateSuccessor(agentIndex, act)
            score = self.value(successor_game_state, agentIndex+1, 0)
            if score > bestScore:
                bestScore = score
                best_action = act
        return best_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    INF = float("inf")
    score = 0;

    food_dists = [0]
    # Better if close to the food, reversely correlated with distance to food
    for food in newFood.asList():
        # Euclidean distance
        food_dist = math.sqrt(math.pow((newPos[0] - food[0]), 2) + math.pow((newPos[1] - food[1]), 2))
        food_dists.append(food_dist)

    if len(food_dists) > 1:
        score += min(food_dists) * 60  # i picked this as a weight
        score -= 20* len(food_dists)

    # Better if the ghosts are scared more:
    for time in newScaredTimes:
        score += time * 10

    for ghostState in newGhostStates:
        ghost_loc = ghostState.getPosition()
        # print(ghost_loc)
        dist = math.sqrt(math.pow((newPos[0] - ghost_loc[0]), 2) + math.pow((newPos[1] - ghost_loc[1]), 2))
        ghost_dists = [0]
        if ghostState.scaredTimer == 0:  # strategy to ghost depends on if its scared
            if dist == 0:
                # definitely avoid meeting a ghost
                return -INF * 10

            if dist > 5:  # its ok if its safely far
                score -= 20*dist
        else:
            ghost_dists.append(dist)

    score -= max(ghost_dists) * (10)

    # print("score ", score)
    return score + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction
