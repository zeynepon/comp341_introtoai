# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0.01  # I decreased the noise.
    return answerDiscount, answerNoise

def question3a():
    answerDiscount = 0.2
    answerNoise = 0.01
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'
    # Negative living reward, so the agent looks for the nearest terminal state
    # Discount not too low, noise low so it risks the cliff

def question3b():
    answerDiscount = 0.2
    answerNoise = 0.2
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'
    # Negative living reward, so the agent looks for the nearest terminal state
    # Discount not too low, noise high so it avoids the cliff

def question3c():
    answerDiscount = 0.9
    answerNoise = 0.01
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'
    # No reward for living, low discount so it can afford to choose the long way
    # (discount rate is high but it will cause lower discount so values will not converge as fast)

def question3d():
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'
    # Same as c, but it avoids the cliff with high noise

def question3e():
    answerDiscount = 1
    answerNoise = 0.25
    answerLivingReward = 10
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    answerEpsilon = 0
    answerLearningRate = 0
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'
    # At epsilon = 0, it circles between the start state and the next
    # square because it is too cautious
    # With other epsilons, it gets close but never actually reaches the terminal state

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
