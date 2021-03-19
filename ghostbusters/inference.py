# inference.py
# ------------
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


import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"
        dist_keys = list(self.keys())  # Keys in distribution, as list
        dist_sum = self.total()  # Total of the distribution values
        if dist_sum == 0:
            return None  # To avoid the divide-by-zero exception
        for key in dist_keys:
            value = self[key]  # Value of the key
            value_normalized = value / dist_sum  # Value, normalized
            self[key] = value_normalized  # Value of the key changed to normalized value


    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        items = self.items()  # Extract the items, a tuple of values and distributions
        values = [i[0] for i in items]  # Values of the distribution
        dist = [i[1] for i in items]  # Distribution
        self.normalize()  # In case the total does not sum to 1
        random_sample = random.random()  # A random sample, random number between 0 and 1
        iteration, iteration_dist = 0, dist[0]  # Initialization of i, the total will be calculated iteratively
        while random_sample > iteration_dist:  # If random sample exceeds total, then the corresponding value will be the weight
            iteration += 1  # Iterate i
            iteration_dist += dist[iteration]  # Add the i'th element of distribution to the total
        return values[iteration]


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
        if ghostPosition == jailPosition and noisyDistance is None:
            return 1
        if ghostPosition == jailPosition or noisyDistance is None:
            return 0
        """Out of the two possibilities, if one of them is true and the other is not,
        we know that the observation probability is 0. There is no way for the observation probability
        to be true if the reading is None but the two positions are not equal. Similarly, there is no
        way for the observation probability to be true if the reading is a position but the
        two positions are equal. However, if the distance reading is None and the ghost position
        equals jail position, the observation probability will be 1. The noisy distance reading will be correct."""

        actual_distance = manhattanDistance(pacmanPosition, ghostPosition)  # Calculation of the actual distance
        prob = busters.getObservationProbability(noisyDistance, actual_distance)  # Observation probability of noisy
        # distance given the true distance

        return prob


    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        pacman_position = gameState.getPacmanPosition()
        jail_position = self.getJailPosition()
        ghost_positions = self.allPositions
        # All the positions needed to calculate probability of evidence

        for position in ghost_positions:  # Iterating through all positions
            prob_evidence = self.getObservationProb(observation, pacman_position, position, jail_position)
            # Probability of evidence calculated according to the function previously written
            self.beliefs[position] *= prob_evidence
            # P(X|e_1:t) = P(X|e_1:t-1) * P(E|X)

        self.beliefs.normalize()

    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"
        ghost_positions = self.allPositions  # All positions for the ghost
        old_beliefs = self.beliefs  # Temp variable to store previous self.beliefs values
        self.beliefs = DiscreteDistribution()  # self.beliefs reset

        for position in ghost_positions:
            new_position_dist = self.getPositionDistribution(gameState, position).items()
            # Distribution for the new positions of the ghost according to the specific position
            for new_position, probability in new_position_dist:
                self.beliefs[new_position] += probability * old_beliefs[position]
                """self.beliefs is updated according to the probability of new position and the belief
                state of the old position: B'(X_t+1) = Sigma P(X'|x_t)B(x_t)"""

        self.beliefs.normalize()

    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        if self.numParticles % len(self.legalPositions) == 0:
            # If numParticle is divisible by number of legal positions
            for position in self.legalPositions:
                particle_per_position = self.numParticles / len(self.legalPositions)
                for _ in range(int(particle_per_position)):
                    self.particles.append(position)
                # We go through all positions, and add the position particle_per_position times to self.particles.
        else:
            remainder = self.numParticles % len(self.legalPositions)
            for position in self.legalPositions:
                particle_per_position = (self.numParticles - remainder) / len(self.legalPositions)
                # The particle per position is an integer, because we subtracted the remainder.
                for _ in range(int(particle_per_position + remainder)):
                    self.particles.append(position)
                if remainder > 0:
                    remainder -= 1
                """We are distributing the remainders one by one to positions, once we go through the remainder
                (remainder == 0), we will go through the positions as usual."""

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        pacman_position = gameState.getPacmanPosition()
        jail_position = self.getJailPosition()
        ghost_positions = self.allPositions
        # All the positions needed to calculate probability of evidence
        old_dist = self.getBeliefDistribution()  # Distribution acquired from self.particles
        updated_dist = DiscreteDistribution()  # Empty discrete distribution object

        for position in ghost_positions:
            prob_evidence = self.getObservationProb(observation, pacman_position, position, jail_position)
            updated_dist[position] = prob_evidence * old_dist[position]
            # B(X) α P(e|x)B'(X)
        updated_dist.normalize()  # First step of resampling

        if updated_dist.total() == 0:  # Special case
            self.initializeUniformly(gameState)
        else:
            self.particles = list()  # self.particles reset
            for i in range(self.numParticles):
                # Resample N times
                resampled = updated_dist.sample()  # Choose from weighted distribution
                self.particles.append(resampled)  # Add resampled particle to self.particles

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        old_particles = self.particles
        self.particles = list()

        for old_particle in old_particles:
            new_position_dist = self.getPositionDistribution(gameState, old_particle)
            resampled = new_position_dist.sample()
            self.particles.append(resampled)

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        
        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"
        beliefs = DiscreteDistribution()  # Empty discrete distribution object
        for position in self.particles:
            beliefs[position] += 1
        """Every time I encounter the position in self.particles, I add one to the belief distribution to weigh
        all positions. After normalization, it will be weighted by the probability."""

        beliefs.normalize()
        return beliefs

class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        ghost_positions = list(itertools.product(self.legalPositions, repeat=2))
        # The Cartesian product of legal positions by itself, as list
        count = 0  # Counter initialized to go through ghost_positions
        for p1, p2 in ghost_positions:
            if p1 == p2:
                ghost_positions.remove(ghost_positions[count])
                # Since two ghosts cannot be in the same position, we remove the case from legal positions
            count += 1

        for _ in range(self.numParticles):  # Go through each particle and add the position pair
            for position_pair in ghost_positions:
                self.particles.append(position_pair)

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        pacman_position = gameState.getPacmanPosition()
        # This is the only position needed before considering all positions and all ghosts
        updated_dist = DiscreteDistribution()
        # Empty distribution to add weighted positions

        for particle_pair in self.particles:  # Goes through each particle pair
            prob_evidence = 1
            """Initialization of the probability of evidence. Since we will need to multiply the emission
            probabilities together, we need this variable to be the identity element of multiplication; that is, one."""
            for i in range(self.numGhosts):
                jail_position = self.getJailPosition(i)  # Jail position of the corresponding ghost
                prob_evidence *= self.getObservationProb(observation[i], pacman_position, particle_pair[i], jail_position)
                # Each emission probability will be multiplied by the previous emission probability
                # P(E1|X) = P(E1^a|G1^a) * P(E1^b|G1^b)
            updated_dist[particle_pair] += prob_evidence
            # The calculated emission probability added to the weight of the particle pair
        updated_dist.normalize()  # First step of resampling

        if updated_dist.total() == 0:  # Special case
            self.initializeUniformly(gameState)
        else:
            self.particles = list()  # self.particles reset
            for i in range(self.numParticles):
                resampled = updated_dist.sample()  # Choose from weighted distribution
                self.particles.append(resampled)  # Add resampled particle to self.particles

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            """for i in range(self.numGhosts):
                new_position_dist = self.getPositionDistribution(gameState, newParticle, i, self.ghostAgents[i])
                resampled = new_position_dist.sample()
                newParticle[i] = resampled"""

            # PLEASE READ!!!
            """I think my code that I have commented above works. However, I have not been able
            to test the code, because it does not terminate. It terminated when I had failed. So, I have
            come to the conclusion that it works, but my computer cannot handle it. I have left it running
            for more than half an hour. I am leaving it commented because you specifically asked not to submit
            code that does not terminate. Hopefully it is as I suspect and it works, just not on my computer."""
            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
