# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from game import Actions
# from searchUtil import mazeDistance, PositionSearchProblem
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
    some Directions.X for some X in the set {North, South, West, East, Stop}
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
    newFood = successorGameState.getFood().asList()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    newGhostPos = successorGameState.getGhostPositions()
    if newPos in newGhostPos: 
        return float('-inf')
    else: 
        newDis = []
        numFood = len(newFood)
        if numFood ==0:
            return float('inf')
        if numFood>0:
            for food in newFood:
                newDis.append(manhattanDistance(newPos,food)+numFood*100)
            fscore =min(newDis)
            gdis2 = 0
            for pos in newGhostPos:
                gdis2+=manhattanDistance(newPos,pos)
            score = gdis2-fscore
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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    # only in minValue, many ghost agent
    def minValue(agentIndex, depth, gameState):
        actions = gameState.getLegalActions(agentIndex)
        numAgents = gameState.getNumAgents()
        if depth == self.depth or len(actions) ==0:
            return self.evaluationFunction(gameState)
        utils = []
        if agentIndex ==numAgents-1:
            for action in actions:
                utils.append(maxValue(0,depth,gameState.generateSuccessor(agentIndex, action)))
        else:
            for action in actions:
                utils.append(minValue(agentIndex+1,depth,gameState.generateSuccessor(agentIndex,action)))
        return min(utils)
            
    def maxValue(agentIndex,depth, gameState):
        actions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(actions) ==0:
            return self.evaluationFunction(gameState)
        utils = []
        for action in actions:
            utils.append(minValue(agentIndex+1,depth+1,gameState.generateSuccessor(agentIndex,action)))
        return max(utils)
    
    
    frontier = {}
    for action in gameState.getLegalActions(0):
        frontier[action] = minValue(1,1,gameState.generateSuccessor(0,action))
    return max(frontier, key = frontier.get)
#     util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    
    def minValue(agentIndex, depth, alpha,beta, gameState):
        v =float('inf')
        betaCur =beta
        actions = gameState.getLegalActions(agentIndex)
        numAgents = gameState.getNumAgents()
        if depth == self.depth or len(actions) ==0:
            return self.evaluationFunction(gameState)
        if agentIndex ==numAgents-1:
            for action in actions:
                v = min(v,maxValue(0,depth, alpha,betaCur,gameState.generateSuccessor(agentIndex,action)))
                if v <alpha:
                    return v
                betaCur  = min(betaCur, v)
        else:
            # how to find the
            for action in actions:
                v = min(v, minValue(agentIndex+1,depth,alpha, betaCur, gameState.generateSuccessor(agentIndex,action)))
                if v < alpha:
                    return v
                betaCur = min(betaCur, v)
        return v
    
    def maxValue(agentIndex,depth, alpha, beta, gameState):
        v = float('-inf')   
        alphaCur = alpha
        actions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(actions) ==0:
            return self.evaluationFunction(gameState)
        for action in actions:
            v= max(v, minValue(agentIndex+1,depth+1,alphaCur,beta, gameState.generateSuccessor(agentIndex,action)))
            if v >beta:
                return v
            alphaCur = alpha
        return v
    alpha = float('-inf')
    beta = float('inf')
    frontier = {}
    for action in gameState.getLegalActions(0):
        v = minValue(1,1,alpha, beta,gameState.generateSuccessor(0,action))
        frontier[action] = v
        if v > beta:
            return action
        alpha = max(alpha, v)
    return max(frontier, key = frontier.get)
    
#     util.raiseNotDefined()

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
    def expValue(agentIndex, depth, gameState):
        actions = gameState.getLegalActions(agentIndex)
        numAgents = gameState.getNumAgents()
        if len(actions) ==0:
            return self.evaluationFunction(gameState)
        totalValue = 0
        p = 1.0/len(actions)
        for action in actions:
            if agentIndex ==numAgents-1:
                v = maxValue(0,depth,gameState.generateSuccessor(agentIndex, action))
            else:
                v=expValue(agentIndex+1,depth,gameState.generateSuccessor(agentIndex, action))
#             print(v)
            totalValue += v*p
        return totalValue
            
    def maxValue(agentIndex,depth, gameState):
        actions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(actions) ==0:
            return self.evaluationFunction(gameState)
        utils = []
        for action in actions:
            utils.append(expValue(agentIndex+1,depth+1,gameState.generateSuccessor(agentIndex,action)))
        return max(utils)
    
    frontier = {}
    for action in gameState.getLegalActions(0):
        frontier[action] = expValue(1,1,gameState.generateSuccessor(0,action))
    return max(frontier, key = frontier.get)
#     util.raiseNotDefined()
# def mazeDistance(point1, point2,currentGameState):
#     frontier = util.Queue()
#     frontier.push((point1,0))
#     reached =[]
#     while frontier.isEmpty()==False:
#         node,total = frontier.pop()
#         if node not in reached:
#             if node ==point2:
#                 return total
#             reached.append(node)
#             successors = []
#             for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
#                 x,y = node
#                 dx, dy = Actions.directionToVector(action)
#                 nextx, nexty = int(x + dx), int(y + dy)
#                 walls = currentGameState.getWalls()
#                 if not walls[nextx][nexty]:
#                     nextState = (nextx, nexty)
#                     successors.append( ( nextState, action, 1) )
#             for c in successors:
#                 state = c[0]
#                 cost = c[2]
#                 frontier.push((state,total+cost))
def euclideanHeuristic(point1, point2, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = point1
    xy2 = point2
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #The evaluation function should evaluate states
    #use the reciprocal of important values (such as distance to food) rather than the values themselves.
    #linear combination of features: compute values for features about the state that you think are important, 
    #and then combine those features by multiplying them by different values and adding the results together. 
    #You might decide what to multiply each feature by based on how important you think it is.
    
    #state
    #use the reciprocal of important values (such as distance to food) rather than the values themselves.
    #linear combination of features: compute values for features about the state that you think are important, 
    #and then combine those features by multiplying them by different values and adding the results together. 
    #You might decide what to multiply each feature by based on how important you think it is.
    foods = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostPositions()
    if currentGameState.getPacmanPosition() in ghosts:
        return float('-inf')
    else:
        if len(foods) ==0:
            return float('inf')
        else:
            score = -len(foods)
#             print('len(food)', len(foods))
#             print('1st score',  8.0/len(foods))
            ghostDis = 0
            mazeDistances = []
            for food in foods:
                dis = manhattanDistance(currentGameState.getPacmanPosition(), food)
                mazeDistances.append(dis)
            score +=1.0/min(mazeDistances)
#             print('2nd score', 5.0/min(mazeDistances))
            ghostDis =0
            ghostDis = min([manhattanDistance(currentGameState.getPacmanPosition(),ghost) for ghost in ghosts])
            score -=1.0/ghostDis
#             print('3rd score', -100.0/ghostDis)
            return score

#     util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

# class ContestAgent(MultiAgentSearchAgent):
#  """
#     Returns an action.  You can use any method you want and search to any depth you want.
#       Just remember that the mini-contest is timed, so you have to trade off speed and computation.

#       Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
#       just make a beeline straight towards Pacman (or away from him if they're scared!)
#   """

#   def getAction(self, gameState):
#     """
#       Returns the minimax action using self.depth and self.evaluationFunction
#     """
#     "*** YOUR CODE HERE ***"
#     util.raiseNotDefined()

