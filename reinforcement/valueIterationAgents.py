# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()  # All states for which we can place a value
        for i in range(self.iterations):  # We will iterate self.iteration times
            prev_values = self.values.copy()  # Batch version, not online version: We need to store V_t
            for state in states:  # Going through all the states in each step
                actions = self.mdp.getPossibleActions(state)  # All possible actions for the corresponding state
                temp_values = util.Counter()  # Temp value counter to store values without changing self.values
                for action in actions:
                    transition_dict = self.mdp.getTransitionStatesAndProbs(state, action)  # Transition model
                    for successor, probability in transition_dict:
                        reward = self.mdp.getReward(state, action, successor)  # Reward for the action
                        current_value = prev_values[successor]  # V_t(s')
                        temp_values[action] += probability * (reward + self.discount * current_value)
                        # for s', P(s'|s,a) * (R(s,a,s') + gamma*(V_t(s')))
                policy = temp_values.argMax()  # best action corresponding to the values calculated above
                self.values[state] = temp_values[policy]  # value of the corresponding action

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q = 0  # Initialization of the Q-value
        transition_dict = self.mdp.getTransitionStatesAndProbs(state, action)  # Transition model
        for successor, probability in transition_dict:
            reward = self.mdp.getReward(state, action, successor)  # Reward for the action
            value = self.values[successor]  # Q_k(s',a')
            q += probability * (reward + self.discount * value)
            # for s, P(s'|s,a) * (R(s,a,s') + gamma*(V_t(s'))), it says to calculate
            # from the value in self.values
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)  # All possible actions
        if actions is None:
            return None
            # If there are no legal actions

        temp_values = util.Counter()  # Temp value counter to store values without changing self.values
        for action in actions:
            q = self.getQValue(state, action)
            temp_values[action] = q
            # Associates the action with the corresponding Q-value

        policy = temp_values.argMax()  # Best action according to the values
        return policy

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()  # All states for which we can place a value
        for i in range(self.iterations):  # We will iterate self.iteration times
            state_index = i % len(states)
            # Remainder of the iteration index; if it exceeds len(states) it will go back to 0
            current_state = states[state_index]  # *state_index*'th state
            if self.mdp.isTerminal(current_state):
                continue  # If the state is terminal, we go through with the next iteration
            temp_values = util.Counter()  # Temp value counter to store values without changing self.values
            actions = self.mdp.getPossibleActions(current_state)  # All possible actions for the corresponding state
            for action in actions:
                temp_values[action] = self.computeQValueFromValues(current_state, action)
                # Calculated the Q-value according to the function written in ValueIterationAgent, which is a superclass
            policy = temp_values.argMax()  # Action with the highest value
            self.values[current_state] = temp_values[policy]  # Value for the state is the value of the policy

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()  # All states
        predecessors = collections.defaultdict(set)  # Empty dictionary for the predecessors, values are a set
        for state in states:
            actions = self.mdp.getPossibleActions(state)  # All possible actions for a state
            for action in actions:
                transition_dict = self.mdp.getTransitionStatesAndProbs(state, action) # Transition dictionary
                for successor, probability in transition_dict:
                    if probability > 0:  # Nonzero probability
                        predecessors[successor].add(state)  # State added to the successor key's value set

        pq_state = util.PriorityQueue()  # Empty priority queue to store states according to diff variable
        for state in states:
            if not self.mdp.isTerminal(state):  # Done for all states but terminal
                highest_q = self.compute_highest_q(state)  # Highest Q-value for the corresponding state
                diff = abs(highest_q - self.values[state])
                # Difference between highest-Q value and the current state value
                pq_state.push(state, -diff)  # State stored with priority -diff

        for i in range(self.iterations):
            if pq_state.isEmpty():  # If no states pushed, terminate
                return None
            popped_state = pq_state.pop()  # Popped state from priority queue
            if not self.mdp.isTerminal(popped_state):
                # Popped state is definitely not terminal as we have pushed non-terminal states, but due diligence
                highest_q = self.compute_highest_q(popped_state)  # Highest Q-value of the popped state
                self.values[popped_state] = highest_q  # Value of popped state updated
                state_predecessors = predecessors[popped_state]  # All predecessors of the popped state
                for predecessor in state_predecessors:
                    highest_q = self.compute_highest_q(predecessor)  # Highest Q-value of the predecessor state
                    diff = abs(highest_q - self.values[predecessor])
                    # Difference between highest-Q value and the predecessor state value
                    if diff > self.theta:
                        pq_state.update(predecessor, -diff)
                        # If higher than threshold, pq_state predecessor priority is updated
                        # Update is used for items already in priority queue

    def compute_highest_q(self, state):
        # A function to compute the highest Q-value for a given state's actions
        actions = self.mdp.getPossibleActions(state)  # All legal actions for a state
        highest_q = -99999  # Initialization of the highest Q-valye
        for action in actions:
            current_q = self.computeQValueFromValues(state, action)
            if current_q > highest_q:
                highest_q = current_q
            # Go through all actions and find the corresponding Q(s,a), if it's higher than highest Q it is
            # the new highest Q
        return highest_q
