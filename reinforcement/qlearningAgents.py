# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.q = util.Counter()  # Just like how we initialized values in value iteration

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        q_value = self.q[(state, action)]  # Similar to getting value in value iteration
        return q_value

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)  # All legal actions
        if actions is None:  # If empty, means it's a terminal state
            return 0.0

        q_pairs = util.Counter()  # Empty counter to store value and action pairs
        for action in actions:
            q_pairs[action] = self.getQValue(state, action)  # Pretty self-explanatory

        max_action = q_pairs.argMax()  # argmax of the q_pairs counter
        return q_pairs[max_action]
        # Value corresponding to the best action

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if actions is None:
            return 0.0

        q_pairs = util.Counter()
        for action in actions:
            q_pairs[action] = self.getQValue(state, action)
        # Everything is the same until here with the computeValue function
        max_action = self.max_tiebreak(q_pairs)
        # Determines the best action by resolving tiebreaks randomly
        return max_action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if legalActions is None:  # No legal actions, terminal state
            return None

        greed = util.flipCoin(self.epsilon)  # Determines if random or policy action with probability epsilon
        if greed:
            actions_list = list(legalActions)
            action = random.choice(actions_list)  # Random action chosen from the action list
        else:
            action = self.computeActionFromQValues(state)  # Best policy computed and implemented

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        next_value = self.computeValueFromQValues(nextState)  # Q(s',a')
        sample = reward + self.discount * next_value  # R(s,a,s') + gamma*Q(s',a')
        self.q[(state, action)] += self.alpha * (sample - self.q[(state, action)])
        # Q(s,a) <- Q(s,a) + alpha * (sample - Q(s,a))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def max_tiebreak(self, q_pairs):
        # I wrote this function to decide on tiebreaks in choosing the best action
        policy = None  # Initialization of policy
        max_value = -99999  # Initialization of value corresponding to policy
        tiebreak = list()  # Empty list in case there is a tiebreak
        for action in q_pairs:
            if q_pairs[action] > max_value:
                max_value = q_pairs[action]
                policy = action
                del tiebreak[:]
            # If the current value is higher than max_value, it is the new max_value
            # We also reset the tiebreak list
            elif q_pairs[action] == max_value:
                tiebreak.append(action)
            # If there is a tiebreak, we will choose randomly from the list

        if len(tiebreak) != 0:
            policy = random.choice(tiebreak)
            # Random choice if tiebreak list is not empty
        return policy


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        feature_vector = self.featExtractor.getFeatures(state, action)  # All the features as a vector
        q = feature_vector * self.weights  # Dot product

        return q

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)  # All features
        current_q = self.getQValue(state, action)  # Q-value of the current state and action
        next_q = self.computeValueFromQValues(nextState)  # Value of the next state
        difference = (reward + self.discount*next_q) - current_q  # Difference variable
        for feature in features:
            self.weights[feature] += self.alpha * difference * features[feature]
            # w_i <- w_i + alpha * difference * f_i(s,a)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
