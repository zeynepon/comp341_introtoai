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
        food_list = newFood.asList()  # Food states are turned into a list
        food_distance = 99999
        ghost_distance = 99999
        # I will find the smallest food distance and ghost distance.

        # I did not want to penalize the 'Stop' action because it does not stop so much with this evaluation function.

        for food in food_list:
            dist = manhattanDistance(newPos, food)
            if dist < food_distance:
                food_distance = dist  # If the current distance is smaller, it is my new food distance

        for ghost in newGhostStates:
            dist = manhattanDistance(newPos, ghost.getPosition())
            if dist < ghost_distance:
                ghost_distance = dist  # If the current distance is smaller, it is my new ghost distance

        eval_reflex = (10 / food_distance) - (1 / (ghost_distance + 0.00001)) + successorGameState.getScore()
        # I am using the reciprocal values for the distances because it is more consistent.
        # Ghost distance may be 0, so I add 0.00001 to it to eliminate any operational errors.
        """When food_distance and ghost_distance have the same weight, Pacman becomes very reluctant
        about the food that is actually near him. I wanted Pacman to move even if the ghost was a little close,
        because otherwise he takes no risks and eventually loses or wins in a long time. To realize this, I
        increased the weight of the food_distance function. food_distance has weight 10 and ghost_distance 1."""
        """If we do not take the score into account no evaluation value works well.
        The score is the main evaluation value and we use the other values for improvement."""

        for scared_timer in newScaredTimes:
            if scared_timer > 0 and ghost_distance == 1:
                # This means that the ghosts are currently scared and if Pacman eats them, they will become unscared.
                # If this happens, I will return a big negative value so Pacman avoids it
                return -99999

        return eval_reflex


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
        minimax, action = self.minimax_value(gameState, 0, 0)
        # minimax_value returns a tuple of minimax and the best action, we return the best action.
        # It is a tuple because the minimax value is used throughout the code.
        return action

    def minimax_value(self, gameState, agent_index, minimax_depth):
        terminal_state = gameState.isWin() \
                         or gameState.isLose() \
                         or minimax_depth == self.depth * gameState.getNumAgents()
        # We know we have reached a terminal state when any of these occurs
        """When I simply wrote minimax_depth == self.depth, it did not cover enough ground because
        the total depth is the "depth" each agent covers times the number of agents. This is due to
        the fact that I increase the depth each time the value of Pacman or the ghosts' move are calculated."""

        if agent_index == gameState.getNumAgents():
            agent_index = 0  # When we go through Pacman & all the ghosts, we return to Pacman's index
            # At this point, we have gone through each agent

        if terminal_state:
            utility = (self.evaluationFunction(gameState), '')
            return utility  # The utility of the state is returned
            # I returned a tuple because the pacman_value & ghost_value functions return a tuple
        elif agent_index == 0:  # Means Pacman is the next agent
            agent_val, minimax_action = self.pacman_value(gameState, agent_index, minimax_depth + 1)
        else:  # agent_index >= 1, it's a ghost's turn
            agent_val, minimax_action = self.ghost_value(gameState, agent_index, minimax_depth + 1)

        return agent_val, minimax_action

    def pacman_value(self, gameState, agent_index, minimax_depth):  # Max value finder
        value = -99999
        actions = gameState.getLegalActions(agent_index)  # All the actions that Pacman can take
        pacman_action = ''  # Empty variable for the best action

        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            successor_value = self.minimax_value(successor, agent_index + 1, minimax_depth)
            # Generated a successor and its value
            temp_value = value
            """The temporary value checks if the max value has changed. If it has, we establish the current
            action as the best action."""
            value = max(value, successor_value[0])
            # minimax_value returns a tuple of value and actions, so I take the 0th element of the tuple.
            if value != temp_value:
                pacman_action = action

        return value, pacman_action

    def ghost_value(self, gameState, agent_index, minimax_depth):  # Min value finder
        value = 99999  # Initialized value is a big variable that will decrease later
        actions = gameState.getLegalActions(agent_index)  # All the legal actions for the ghost to take
        ghost_action = ''  # Empty variable for the best action

        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            successor_value = self.minimax_value(successor, agent_index + 1, minimax_depth)
            # The successor and its value is generated
            temp_value = value
            """The temporary value checks if the max value has changed. If it has, we return the current
            action as the best action."""
            value = min(value, successor_value[0])
            # minimax_value returns a tuple of value and actions, so I take the 0th element of the tuple.
            if value != temp_value:
                ghost_action = action

        return value, ghost_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha_beta, action = self.alpha_beta_value(gameState, 0, 0, float('-inf'), float('inf'))
        """alpha_beta_value returns a tuple of the alpha-beta value
        and the best action, we return the best action."""
        return action

    def alpha_beta_value(self, gameState, agent_index, alpha_beta_depth, alpha, beta):
        terminal_state = gameState.isWin() \
                         or gameState.isLose() \
                         or alpha_beta_depth == self.depth * gameState.getNumAgents()
        # We know we have reached a terminal state when any of these occurs

        if agent_index == gameState.getNumAgents():
            agent_index = 0  # Means we have gone through all of the agents and we go back to Pacman

        if terminal_state:
            utility = (self.evaluationFunction(gameState), '')  # The utility of the terminal state is returned
            return utility
        elif agent_index == 0:  # Means it's Pacman's turn
            ab_val, ab_action = self.pacman_value(gameState, agent_index, alpha_beta_depth + 1, alpha, beta)
        else:  # Means agent_index >= 1, so it's a ghost's turn
            ab_val, ab_action = self.ghost_value(gameState, agent_index, alpha_beta_depth + 1, alpha, beta)
        # I did not want to make the line so long, so I used the abbreviation ab as alpha-beta

        return ab_val, ab_action

    def pacman_value(self, gameState, agent_index, alpha_beta_depth, alpha, beta):  # Max value finder
        value = -99999
        actions = gameState.getLegalActions(agent_index)  # All the actions that Pacman can take
        pacman_action = ''  # Empty variable for the best action

        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            successor_value = self.alpha_beta_value(successor, agent_index + 1, alpha_beta_depth, alpha, beta)
            # The successor and its value is generated.
            temp_value = value
            """ Temporary value variable to see if the value has changed
            so that the best action is the one corresponding to the value."""
            value = max(value, successor_value[0])
            # alpha_beta_value returns a tuple of value and actions, so I take the 0th element of the tuple.
            if value != temp_value:
                pacman_action = action  # If the value has changed, this is the corresponding action
            if value > beta:
                beta = float('inf')
                """As the assignment pdf suggests, beta changes for different nodes so
                it should not be the same for the whole search. Therefore, I reset it to inf."""
                return beta, pacman_action
                # If the value is greater than the previously chosen max value, I can prune the other successors.
            alpha = max(alpha, value)

        return value, pacman_action

    def ghost_value(self, gameState, agent_index, alpha_beta_depth, alpha, beta):  # Min value finder
        value = 99999
        actions = gameState.getLegalActions(agent_index)  # All the actions that the ghost can take
        ghost_action = ''  # Empty variable for the best action

        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            successor_value = self.alpha_beta_value(successor, agent_index + 1, alpha_beta_depth, alpha, beta)
            # The successor and its value is generated.
            temp_value = value
            """ Temporary value variable to see if the value has changed
            so that the best action is the one corresponding to the value."""
            value = min(value, successor_value[0])
            # alpha_beta_value returns a tuple of value and actions, so I take the 0th element of the tuple.
            if value != temp_value:
                ghost_action = action  # If the value has changed, this is the corresponding action
            if value < alpha:
                alpha = float('-inf')
                """As the assignment pdf suggests, alpha changes for different nodes so
                it should not be the same for the whole search. Therefore, I reset it to -inf."""
                return alpha, ghost_action
            # If the value is smaller than the previously chosen min value, I can prune the other successors.
            beta = min(beta, value)

        return value, ghost_action


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
        expectimax, action = self.expectimax_value(gameState, 0, 0)
        """expectimax_value returns a tuple of the expectimax value and the corresponding action,
        we return the action."""
        return action

    def expectimax_value(self, gameState, agent_index, expectimax_depth):
        terminal_state = gameState.isWin() \
                         or gameState.isLose() \
                         or expectimax_depth == self.depth * gameState.getNumAgents()
        # We know we have reached a terminal state when any of these occurs

        if agent_index == gameState.getNumAgents():
            agent_index = 0
            # Similar to the previous algorithms, this means that we have gone through each agent

        if terminal_state:
            utility = (self.evaluationFunction(gameState), '')
            return utility  # In case of the terminal state, the utility of the state is returned.
        elif agent_index == 0:  # Means it is Pacman's turn
            ex_val, ex_action = self.pacman_value(gameState, agent_index, expectimax_depth + 1)
        else:  # agent_index >= 1, means it is the stochastic ghosts' turn
            ex_val, ex_action = self.stochastic_value(gameState, agent_index, expectimax_depth + 1)
        # We do not need a min value finder here because we are no longer dealing with adversarial agents

        return ex_val, ex_action

    def pacman_value(self, gameState, agent_index, expectimax_depth):  # Max value finder
        value = -99999  # Initialization of the max value
        actions = gameState.getLegalActions(agent_index)  # All the legal actions that Pacman can take
        pacman_action = ''  # Empty function for the best action

        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            successor_value = self.expectimax_value(successor, agent_index + 1, expectimax_depth)
            # Successor and its value are generated, as usual
            temp_value = value  # Temp value to see if the best action has changed
            value = max(value, successor_value[0])  # Max of the previously obtained value and the successor's value
            if value != temp_value:
                pacman_action = action  # Best action is now this action

        return value, pacman_action

    def stochastic_value(self, gameState, agent_index, expectimax_depth):  # Stochastic value finder
        value = 0  # Initialized as 0 because we will be adding to it later
        actions = gameState.getLegalActions(agent_index)  # All the legal action that the ghost can take
        probability_action = 1 / len(actions)
        # All the actions are uniformly distributed, so the expectation of each action is 1/# of actions

        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            successor_value = self.expectimax_value(successor, agent_index + 1, expectimax_depth)
            # Successor and its value are generated
            value += probability_action * successor_value[0]
            """We add to the value the successor value with the weight of its probability.
            Since this is how we calculate the expectation of a random variable, the value
            becomes the expected value of the ghost."""

        return value, ''
        # No need for the action here, but a tuple is required for the entirety of the code.


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Detailed description is in the report :)
    """
    "*** YOUR CODE HERE ***"
    food_distance = 99999
    ghost_distance = 99999
    # Initialization to the variables that store the minimum distance to food and ghosts.

    pacman_position = currentGameState.getPacmanPosition()  # Current Pacman position
    all_food = (currentGameState.getFood()).asList()  # List of all food positions
    ghosts = currentGameState.getGhostStates()  # All ghost positions

    for food in all_food:
        dist = manhattanDistance(pacman_position, food)  # Distance from Pacman to food
        if dist < food_distance:
            food_distance = dist  # At the end, becomes the minimum distance from Pacman to food

    for ghost in ghosts:
        dist = manhattanDistance(pacman_position, ghost.getPosition())  # Distance from Pacman to the ghost
        if dist < ghost_distance:
            ghost_distance = dist  # At the end, becomes the minimum distance from Pacman to the ghost

    better_eval = 100 * currentGameState.getScore() + (50 / food_distance) - (1 / (ghost_distance + 0.00001))
    """When I used the same weight of the food distance relative to the ghost distance, I ran out of time.
    This has to do with the fact that we are not dealing with actions, but only the current state. It avoided
    the ghost really well, but stopped until the ghost came too close and was not interested in the food at all.
    So I changed the weights of the score and the food distance."""

    return better_eval

# Abbreviation
better = betterEvaluationFunction
