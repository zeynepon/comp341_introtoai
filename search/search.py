# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    root = problem.getStartState() # Get the start state

    visited = set()  # We will add expanded states to this set to avoid expanding twice
    frontier = util.Stack()  # We use stack to implement LIFO
    frontier.push((root, []))  # Node + path we took to get to the node

    while not frontier.isEmpty():
        node, path = frontier.pop()  # Pop frontier to get the node & path
        if problem.isGoalState(node):
            return path  # Finish code if goal is reached
        visited.add(node)  # Add node to visited set
        successors = problem.getSuccessors(node)  # Get the possible next states, LIFO
        for node, action, _ in successors:
            if node not in visited:
                frontier.push((node, path + [action]))  # Push the states with the action added to the path

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    root = problem.getStartState()  # Get the start state

    visited = set()  # We will add expanded states to this set to avoid expanding twice
    frontier = util.Queue()  # We use queue to implement FIFO
    frontier.push((root, []))  # Node + path we took to get to the node
    visited.add(root)  # To ensure start state is not repeated

    while not frontier.isEmpty():
        node, path = frontier.pop()  # Pop frontier to get the node & path
        if problem.isGoalState(node):
            return path  # Finish code if goal is reached
        successors = problem.getSuccessors(node)  # Get the possible next states, FIFO
        for node, action, _ in successors:
            if node not in visited:
                visited.add(node)  # Addition of node onto the visited set
                frontier.push((node, path + [action]))  # Push the states with the action added to the path


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    root = problem.getStartState()  # Start state
    visited = set()  # Empty set to record expanded states
    frontier = util.PriorityQueue()  # We implement UCS with a priority queue
    frontier.push((root, []), 0)  # Cost of start state is 0

    while not frontier.isEmpty():
        node, path = frontier.pop()  # Pop frontier to get to the current node & path
        if problem.isGoalState(node):
            return path  # Finish code if goal is reached
        cost = problem.getCostOfActions(path)
        """problem.getCostOfActions returns the cost of the current path.
            We store the obtained cost in the *cost* variable."""

        if node not in visited:
            """We need to determine the successors inside the if statement to ensure that no successor
                gets expanded twice."""
            visited.add(node)  # Addition of current node onto the visited set
            successors = problem.getSuccessors(node)  # Determination of successors
            for node, action, _ in successors:
                if problem.getCostOfActions(path + [action]) > cost:
                    frontier.update((node, path + [action]), problem.getCostOfActions(path + [action]))
                else:
                    frontier.push((node, path + [action]), cost)
                # If the child node's priority is higher than the parent node, the frontier node is updated.

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    root = problem.getStartState()  # Start state
    root_heuristic = heuristic(root, problem)  # Heuristic value of the root function
    f_n = root_heuristic  # Priority function, cost of start state is 0 so it is not included
    visited = set()  # Empty set to record expanded states
    frontier = util.PriorityQueue()  # We implement A* search with a priority queue
    frontier.push((root, []), f_n)

    while not frontier.isEmpty():
        node, path = frontier.pop()  # Pop frontier to get to the current node & path
        if problem.isGoalState(node):
            return path  # Finish code if goal is reached
        if node not in visited:
            visited.add(node)  # Addition of current node onto the visited set
            successors = problem.getSuccessors(node)  # Determination of successors
            for node, action, _ in successors:
                node_heuristic = heuristic(node, problem)  # Heuristic value of the current node
                f_n = problem.getCostOfActions(path + [action]) + node_heuristic
                frontier.push((node, path + [action]), f_n)
                # The last argument depicts the priority function.



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
