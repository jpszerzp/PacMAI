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
import random, util, sys

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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return ManhattanAgent(currentGameState) # TODO: put this back: successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return ManhattanAgent(currentGameState) # TODO: replace back with this -> currentGameState.getScore()

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
    
    """
        Implementation of agent with minimax strategy.
    """

    def minimax_recurse(self, gameState, depth, max_mode = True, ghost_agent_id = 1):
      if depth == 0 or gameState.isWin() or gameState.isLose():
        return None, self.evaluationFunction(gameState)
      comp_number = -float('inf') if max_mode else float('inf') 
      desired_actions = []
      if max_mode:
        for action_taken in gameState.getLegalActions():
          det_score = self.minimax_recurse(gameState.generateSuccessor(0, action_taken), depth-1, not max_mode)[1]
          if (comp_number < det_score):
            comp_number = det_score
            desired_actions = [action_taken]
          elif (comp_number == det_score):
            desired_actions.append(action_taken)
      else:
        if (ghost_agent_id < gameState.getNumAgents() - 1):
          for action_taken in gameState.getLegalActions(ghost_agent_id):
            det_score = self.minimax_recurse(gameState.generateSuccessor(ghost_agent_id, action_taken), depth, max_mode, ghost_agent_id + 1)[1]
            if (comp_number > det_score):
              comp_number = det_score
              desired_actions = [action_taken]
            elif (comp_number == det_score):
              desired_actions.append(action_taken)
        elif (ghost_agent_id == gameState.getNumAgents() - 1):
          for action_taken in gameState.getLegalActions(ghost_agent_id):
            det_score = self.minimax_recurse(gameState.generateSuccessor(ghost_agent_id, action_taken), depth-1, not max_mode)[1]
            if (comp_number > det_score):
              comp_number = det_score
              desired_actions = [action_taken]
            elif (comp_number == det_score):
              desired_actions.append(action_taken)
        else:
          print "Ghost AgentID obtained is larger than the number of ghosts in minimax_recurse"
          sys.exit(1)
      return desired_actions, comp_number

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
        """
        return random.choice(self.minimax_recurse(gameState, self.depth*2)[0])

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    
    """
        Optimized version with alpha-beta pruning.
    """

    def alpha_beta_recurse(self, gameState, depth, min_lim, max_lim, max_mode = True, ghost_agent_id = 1):
      if depth == 0 or gameState.isWin() or gameState.isLose():
        return None, self.evaluationFunction(gameState)
      comp_number = min_lim if max_mode else max_lim
      backup_so_far = -float('inf') if max_mode else float('inf')
      desired_actions = None
      if max_mode:
        for action_taken in gameState.getLegalActions():
          det_score = self.alpha_beta_recurse(gameState.generateSuccessor(0, action_taken), depth-1, comp_number, max_lim, not max_mode)[1]
          if (det_score > comp_number):
            comp_number = det_score
            desired_actions = [action_taken]
          elif (det_score == comp_number) and desired_actions != None:
            desired_actions.append(action_taken)
          else:
            backup_so_far = max(backup_so_far, det_score)
          if det_score > max_lim:
            return None, det_score # max_lim
      else:
        if (ghost_agent_id < gameState.getNumAgents() - 1):
          for action_taken in gameState.getLegalActions(ghost_agent_id):
            det_score = self.alpha_beta_recurse(gameState.generateSuccessor(ghost_agent_id, action_taken), depth, min_lim, comp_number, max_mode, ghost_agent_id + 1)[1]
            if (det_score < comp_number):
              comp_number = det_score
              desired_actions = [action_taken]
            elif (comp_number == det_score) and desired_actions != None:
              desired_actions.append(action_taken)
            else:
              backup_so_far = min(backup_so_far, det_score)
            if det_score < min_lim:
              return None, det_score # min_lim
        elif (ghost_agent_id == gameState.getNumAgents() - 1):
          for action_taken in gameState.getLegalActions(ghost_agent_id):
            det_score = self.alpha_beta_recurse(gameState.generateSuccessor(ghost_agent_id, action_taken), depth-1, min_lim, comp_number, not max_mode)[1]
            if (det_score < comp_number):
              comp_number = det_score
              desired_actions = [action_taken]
            elif (comp_number == det_score) and desired_actions != None:
              desired_actions.append(action_taken)
            else:
              backup_so_far = min(backup_so_far, det_score)
            if det_score < min_lim:
              return None, det_score # min_lim
        else:
          print "Ghost AgentID obtained is larger than the number of ghosts in minimax_recurse"
          sys.exit(1)
      if desired_actions != None:
        return desired_actions, comp_number
      else:
        return desired_actions, backup_so_far

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """ 
        return random.choice(self.alpha_beta_recurse(gameState, self.depth*2, -float('inf'), float('inf'))[0])

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    
    """
        Implementation of expectimax strategy.
    """
    
    def expectimax_recurse(self, gameState, depth, max_mode = True, ghost_agent_id = 1):
      if depth == 0 or gameState.isWin() or gameState.isLose():
        return None, self.evaluationFunction(gameState)
      comp_number = -float('inf') if max_mode else 0 
      desired_actions = []
      num_actions = 0
      if max_mode:
        for action_taken in gameState.getLegalActions():
          det_score = self.expectimax_recurse(gameState.generateSuccessor(0, action_taken), depth-1, not max_mode)[1]
          if (comp_number < det_score):
            comp_number = det_score
            desired_actions = [action_taken]
          elif (comp_number == det_score):
            desired_actions.append(action_taken)
          num_actions += 1
      else:
        if (ghost_agent_id < gameState.getNumAgents() - 1):
          for action_taken in gameState.getLegalActions(ghost_agent_id):
            det_score = self.expectimax_recurse(gameState.generateSuccessor(ghost_agent_id, action_taken), depth, max_mode, ghost_agent_id + 1)[1]
            comp_number += det_score
            num_actions += 1
        elif (ghost_agent_id == gameState.getNumAgents() - 1):
          for action_taken in gameState.getLegalActions(ghost_agent_id):
            det_score = self.expectimax_recurse(gameState.generateSuccessor(ghost_agent_id, action_taken), depth-1, not max_mode)[1]
            comp_number += det_score
            num_actions += 1
        else:
          print "Ghost AgentID obtained is larger than the number of ghosts in minimax_recurse"
          sys.exit(1)
      if max_mode:
        return desired_actions, comp_number
      else:
        return None, float(comp_number)/num_actions

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        action_chosen = random.choice(self.expectimax_recurse(gameState, self.depth*2)[0])
        return action_chosen

def positionFactor(pacmanPos, ghostState):
    LENIENCE = 0.6
    ghostPos = ghostState.getPosition()
    ghostDirection = ghostState.getDirection()
    if ghostDirection == Directions.NORTH:
      return LENIENCE if pacmanPos[1] < ghostPos[1] else 1
    elif ghostDirection == Directions.SOUTH:
      return LENIENCE if pacmanPos[1] > ghostPos[1] else 1
    elif ghostDirection == Directions.WEST:
      return LENIENCE if pacmanPos[0] > ghostPos[0] else 1
    else:
      return LENIENCE if pacmanPos[0] < ghostPos[0] else 1

def ManhattanAgent(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
    """
    
    """
        Manhattan distance heuristics.
    """
    
    if currentGameState.isLose(): 
      return -float("inf")
    elif currentGameState.isWin():
      return float("inf")

    pacman_state = currentGameState.getPacmanState()
    pacman_pos = pacman_state.configuration.getPosition()
    game_score = currentGameState.getScore()

    ghost_states = currentGameState.getGhostStates()
    food_left = currentGameState.getNumFood()
    capsules_left = len(currentGameState.getCapsules())
    dist_closest_food = min(map(lambda tile: manhattanDistance(pacman_pos, tile), currentGameState.getFood().asList()))
    dist_closest_ghost = float('inf')
    dist_closest_scared_ghost = float('inf')
    for gs in ghost_states:
      if gs.scaredTimer:
        dist_closest_scared_ghost = min(dist_closest_scared_ghost, manhattanDistance(pacman_pos, gs.getPosition()))
      else:
        dist_closest_ghost = min(dist_closest_ghost, manhattanDistance(pacman_pos, gs.getPosition()))
    # TODO: Maybe set a minimum/maximum value for distances so that infinity isn't possible
    # TODO: Tweak the weights!
    weights = [1, -1.5, -2, -2, -20, -4]
    indi_scores = [game_score, dist_closest_food, (1./dist_closest_ghost), (1./dist_closest_scared_ghost), capsules_left, food_left]
    score = 0
    for i in range(len(weights)):
      score += weights[i]*indi_scores[i]
    return score

lookdist=2
lookgrids=[]
for i in range(-lookdist, lookdist+1):
    for j in range(-lookdist, lookdist+1):
        if (abs(i)+abs(j)<=lookdist and (i!=0 or j!=0)):
            lookgrids.append((i, j))

class ReinforcementAgent(Agent):
    
    """
        Reinforcement learning strategy (used in as evaluation function like above)
    """
    
    def getAction (self, gameState):
        if (random.random()<=0.1):
            tblState = self.getStateFeatures(gameState)
            action = random.choice(gameState.getLegalActions())
        else:
            tblState = self.getStateFeatures(gameState)
            
            qualities = [self.getQuality(tblState, a) for a in gameState.getLegalActions()]
            maxQuality = max(qualities)
            maxQualityCount = qualities.count(maxQuality)
            if maxQualityCount>1:
                maxQualitiyIndices = [i for i in range(len(gameState.getLegalActions())) if qualities[i]==maxQuality]
                maxQualityIndex = random.choice(maxQualitiyIndices)
            else:
                maxQualityIndex = qualities.index(maxQuality)
            actions = gameState.getLegalActions()
            action = actions[maxQualityIndex]
            
            px, py = gameState.getPacmanPosition()
            reward = 0
            scareT = gameState.getGhostState(random.choice(range(1, gameState.getNumAgents()))).scaredTimer
            
            if self.hasGhost(gameState.getGhostPositions(), px, py):
                if (scareT>0):
                    reward = 0
                    if (self.prevState is not None):
                        self.learn(self.prevState, self.prevAction, reward, tblState)
                elif(scareT<=0):
                    reward = -300
                    if (self.prevState is not None):
                        self.learn(self.prevState, self.prevAction, reward, tblState)
                    self.prevState = None
            
            if gameState.hasFood(px, py):
                reward = 10
                if self.prevState is not None:
                    self.learn(self.prevState, self.prevAction, reward, tblState)
            
            if self.hasCaps(gameState.getCapsules(), px, py):
                reward = 50
                if self.prevState is not None:
                    self.learn(self.prevState, self.prevAction, reward, tblState)
    
        self.prevState = tblState
        self.prevAction = action

        return action

    def getQuality(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQuality(self, state, action, reward, value):
        oldQ = self.q.get((state, action), None)
        if oldQ is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] += 0.2 * (value - oldQ)

    def learn(self, state1, action1, reward, state2):
        maxQ = max([getQuality(state2, a) for a in state2.getLegalActions()])
        self.learnQuality(state1, action1, reward, reward+0.9*maxQ)

    # gameState.getLegalActions() v.s. self.actions?

    def hasGhost(self, gp, x, y):
        for gx, gy in gp:
            if (gx == x and gy == y):
                return True
        return False
    
    def hasCaps(self, c, x, y):
        for cx, cy in c:
            if (cx==x and cy==y):
                return True
        return False

    def getStateFeatures(self, state):
        curX, curY = state.getPacmanPosition()
        foods = state.getFood()
        caps = state.getCapsules()
        ghostPos = state.getGhostPositions()
        hasWall = state.hasWall(curX, curY)

        def gridVal (state, x, y, f, c, gp, w):
            if gp is not None and self.hasGhost(gp, x, y):
                return 4
            elif c is not None and self.hasCaps(c, x, y):
                return 3
            elif f is not None and state.hasFood(x, y):
                return 2
            elif w:
                return 1
            else:
                return 0
        print tuple([gridVal(state, curX+i, curY+j, foods, caps, ghostPos, hasWall) for i, j in lookgrids])
        return tuple([gridVal(state, curX+i, curY+j, foods, caps, ghostPos, hasWall) for i, j in lookgrids])

# Abbreviation
better = ManhattanAgent

