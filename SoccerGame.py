
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 16:03:37 2018

@author: nkieu
TODO:
    write Friend first, it is the easiest
    write Foe
    write CE
Notes:
    Fig 3: error values reflect player A (right)'s Q's values corresponding to state s
    with player A taking action S and player B sticking
    fixed: bot can still be in the same field if 1 Stick and the other one moves
    note: p2 in Greenwald paper, under MarkovGamme = Q(s, a) = (1-gamma) * reward
    
"""

import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

NROW = 2
NCOL = 4
moveName = ["N", "E", "S", "W", "ST"]
moveList = [[-1, 0],
            [0, 1],
            [1, 0],
            [0, -1],
            [0, 0]]

moveRange = range(len(moveList))

class Foe_Solver:
    def maxmin(self, payoff, solver = 'glpk'):
         # with Row players. Notice G is transpose
        # To do: turn this to work with array of arbitrary size
        # aka: Minimax - Foe 
        num_vars = payoff.shape[0]
        
        # minimize c * x
        # c has 1 more elements than A
        # the first element is V
        # Notes: we include V as a variable to run the minimization over
        # the -1 in the beginning means that we are actually maximizing V
        c = [-1] + [0 for i in range(num_vars)]
        c = np.array(c, dtype="float")
        c = matrix(c)
        
        # constraints G*x <= h
        G = np.matrix(payoff, dtype="float").T # reformat each variable im in a row
        G *= -1 # minimization constraint
        G = np.vstack([G, np.eye(num_vars) * -1]) # > 0 constraint for all vars
        new_col = [1 for i in range(payoff.shape[1])] + [0 for i in range(num_vars)]
        G = np.insert(G, 0, new_col, axis=1) # insert utility column
        G = matrix(G)
        
        # constraint for payoff
        # constrait for probability >> 0
        h = ([0 for i in range(payoff.shape[1])] + 
             [0 for i in range(num_vars)])
        h = np.array(h, dtype="float")
        h = matrix(h)
        
        # contraints Ax = b
        A = [0] + [1 for i in range(num_vars)]
        A = np.matrix(A, dtype="float")
        A = matrix(A)
        
        # Note that b is scalar
        b = np.matrix(1, dtype="float")
        b = matrix(b)
        
        sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)
        return list(sol['x'])
    
    def test(self):
        A = [[0.0, 1.02], [-1.02, 0.0], [2.66, -1.03]]
        A= np.array(A)
        
        # Value and probabilities of the row player 
        result = self.maxmin(A)
        result = list(result['x'])
        print("Row player", result)
        
        # change into the payoff of the column player
        A = np.array(A).T*(-1)
        result = self.maxmin(A)
        result = list(result['x'])
        print("Column player", result)
        
class CE_Solver:
    def ce(self, payoff, solver=None):
        self.nrow, self.ncol, _ = payoff.shape
        num_vars = self.nrow * self.ncol
        
        # maximize matrix c
        c = np.ravel(np.sum(payoff, axis = 2)).astype("float")
        # equation 9 in 
#        c = np.array(c, dtype="float")
        c = matrix(c)
        c *= -1 # cvxopt minimizes so *-1 to maximize
        
        # constraints G*x <= h
        G = self.build_ce_constraints(A=payoff)
        G = np.vstack([G, np.eye(num_vars) * -1]) # > 0 constraint for all vars
        h_size = len(G)
        G = matrix(G)
        h = [0 for i in range(h_size)]
        h = np.array(h, dtype="float")
        h = matrix(h)
        
        # contraints Ax = b
        # Sum to 1
        A = [1 for i in range(num_vars)]
        A = np.matrix(A, dtype="float")
        A = matrix(A)
        b = np.matrix(1, dtype="float")
        b = matrix(b)
        sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)
        return sol
    
    def build_ce_constraints(self, A):
        G = []
        # row player
        nrow, ncol, _ = A.shape
        
        for i in range(nrow): # action row i
            for j in range(nrow): # action row j
                # if i == j => nothing
                # this is to handle the case when there are more than 2 actions
                if i != j:
                    constraints = [0 for i in range(nrow*ncol)]
                    
                    # the base strategy will get expected payoff at last as good as other strategy
                    # the signs are flipped to accommodate cvxopt
                    for k in range(ncol):
                        constraints[i*ncol + k] = (- A[i,k][0] 
                                        + A[j,k][0])
                    G += [constraints]
                    
        # col player
        for i in range(ncol): # action column i
            for j in range(ncol): # action column j
                if i != j:
                    constraints = [0 for i in range(nrow*ncol)]
                    for k in range(nrow):
                        constraints[i + (k * ncol)] = (
                            - A[k, i][1]
                            + A[k, j][1])
                    G += [constraints]
        
        return np.matrix(G, dtype="float")
    
    def test(self):
        # Correlated Equilibrium
        A = [[6, 6], [2, 7],[7, 2], [0, 0]]
        A = np.array(A)
        sol = self.ce(A)
        print(sol['x'])
    
    def test_2(self):
        # Correlated Equilibrium
        payoff = [[[6, 6], [2, 7],[7, 2]], 
             [[0, 0], [7, 6], [8, 5]]]
        payoff = np.array(payoff)
        sol = self.ce(payoff)
        print(sol['x'])


class Players:
    def __init__(self, name = None, x=0,y=0, hasBall=False):
        self.x = x
        self.y = y
        self.hasBall = hasBall
        self.name = name
    
    def move(self, thisMove):
        self.x += moveList[thisMove][0]
        self.y += moveList[thisMove][1]
        pass
        
class SoccerGame:
    def __init__(self):
        self.nrow = NROW
        self.ncol = NCOL
        self.players = [Players(name='a'), Players(name='b')]
        self.numPlayers = 2
            
    def getStateID_player(self, player):
        # given player location => get the number of the cell
        # in str
        cellNum = player.x * self.nrow + player.y
        return str(cellNum)
    
    def getStateID(self):
        # given both player's location
        # get the unique ID
        state_player1 = self.getStateID_player(self.players[0])
        state_player2 = self.getStateID_player(self.players[1])
        hasBall = '0' if self.players[0].hasBall else '1'
        return state_player1 + state_player2 + hasBall
    
    def reset(self):
        x1, y1 = np.random.choice(range(self.nrow)), np.random.choice(range(1, self.ncol-1))
        self.players[0].x = x1
        self.players[0].y = y1
        
        # make sure that they are not in the same spot
        x2, y2 = x1, y1
        while x1 == x2 and y1 == y2:
            x2, y2 = np.random.choice(range(self.nrow)), np.random.choice(range(1, self.ncol-1))
        
        self.players[1].x = x2
        self.players[1].y = y2
        
        playerHasBall = np.random.choice(range(2))
        self.players[playerHasBall].hasBall = True
        self.players[self.numPlayers - 1 - playerHasBall].hasBall = False
    
    def reset_default(self):
        self.players[0].x = 0
        self.players[0].y = 1
        self.players[0].hasBall = True
        
        self.players[1].x = 0
        self.players[1].y = NCOL - 2
        self.players[1].hasBall = False
        
    def nextState(self, action):
        # check whether action is legitimate
        collide = self.checkCollide(action)
        
        firstMove = np.random.binomial(1, 0.5)
        secondMove = self.numPlayers - 1 - firstMove
        
        if collide:
            # only the first will move
            # this condition is to avoid collision
            if action[secondMove] != 4:
                self.players[firstMove].move(action[firstMove])
            
            # if the second move has the ball, the ball will change possession
            if self.players[secondMove].hasBall == True:
                self.players[secondMove].hasBall = False
                self.players[secondMove].hasBall = True
        else:
            for i in range(2):
                self.players[i].move(action[i])
            
        p1, p2, done = self.reward()
        
        return p1, p2, done
    
    def checkCollide(self, action):
        x1 = self.players[0].x + moveList[action[0]][0]
        y1 = self.players[0].y + moveList[action[0]][1]
        
        x2 = self.players[1].x + moveList[action[1]][0]
        y2 = self.players[1].x + moveList[action[1]][1]
        
        return x1 == x2 and y1 == y2
        
    def reward(self):
        if self.players[0].hasBall and self.players[0].y == NCOL -1 :
            return 100, -100, True
        elif self.players[0].hasBall and self.players[0].y == 0 :
            return -100, 100, True
        elif self.players[1].hasBall and self.players[1].y == 0:
            return -100, 100, True
        elif self.players[1].hasBall and self.players[1].y == NCOL - 1:
            return 100, -100, True
        else:
            return 0, 0, False
    
    def printState(self):
        for i in range(self.numPlayers):
            print("Player", i+1, 
                  self.players[i].x, 
                  self.players[i].y, 
                  self.players[i].hasBall)
    
    def printState_graph(self):
        output = [['o' for i in range(NCOL)] for j in range(NROW)]
        
        for thisPlayer in self.players:
            if thisPlayer.hasBall:
                output[thisPlayer.x][thisPlayer.y] = thisPlayer.name + 'x'
            else:
                output[thisPlayer.x][thisPlayer.y] = thisPlayer.name
        for i in output:
            print(i)
        
class Qlearning:
    def __init__(self, game, verbose = False):
        self.game = game
        self.verbose = verbose
        self.QV = [{}, {}]
#        self.V = [{}, {}] 
        self.visits = {}
        maxCell = game.nrow * game.ncol - 1
        
        # QV for each player
        # QV = dictionary of State - matrix of action 
        # if there is a probability distribution of action => just take the weighted value
        # to get the value of the state
        for thisPlayer in range(game.numPlayers):
            for p1 in range(maxCell):
                for p2 in range(maxCell):
                    for hasBall in range(2):
                        name = str(p1) + str(p2) + str(hasBall)
                        self.QV[thisPlayer][name] = np.zeros((len(moveList),len(moveList)))
                        self.visits[name] = np.zeros((len(moveList),len(moveList)))
    
    def newAction(self):
        # generate a random action
        return [np.random.choice(self.possibleAction(0)),
                np.random.choice(self.possibleAction(1))]
    
    def possibleAction(self, thisPlayer):
        # get the list of possible Action
        tmp = []
        for i in moveRange:
            x1 = self.game.players[thisPlayer].x + moveList[i][0]
            y1 = self.game.players[thisPlayer].y + moveList[i][1]
                
            if 0 <= x1 and x1 < NROW and 0 <= y1 and y1 < NCOL:
                tmp.append(i)
        return tmp
            
    def getLearningRate(self, n):
        alpha = 0
        if n < 1000:
            alpha = 0.5
        elif n < 2000:
            alpha = 0.1
        elif n < 3000:
            alpha = 0.01
        elif n < 5000:
            alpha = 0.001
        return alpha
    
    def getEpsilon(self, n):
        # Epslion-greedy search
        epsilon = 0
        if n == 0:
            epsilon = 1
        elif n < 100:
            epsilon = 0.20
        elif n < 1000:
            epsilon = 0.10
        elif n < 3000:
            epsilon = 0.05
        return epsilon
    
    def simulate(self, n = 10, method = 'FriendQ', solver = CE_Solver):
        gamma = 0.9
        rewards = [0, 0]
        
        deltaList = []
        
        for thisGame in range(n):
            self.game.reset_default()
            if self.verbose: self.game.printState_graph()
            
            deltaUpdate = 0.0
            currentState = self.game.getStateID()
            move = self.newAction()
            state_prime = None
            done = False
            count = 0
            
            deltaQ = 0.0
            while count < 100 and not done:
                count += 1
                rewards[0], rewards[1], done = self.game.nextState(move)
                state_prime = self.game.getStateID()
                if self.verbose: 
                    print(rewards[0], rewards[1])
                    self.game.printState_graph()
                
                # Update learning rate
                
                self.visits[currentState][move[0], move[1]] += 1
                alpha = max(0.001, 1.0 / self.visits[currentState][move[0], move[1]])
                
                # Get V and next move
                values, nextMove = [0.0, 0.0], [0, 0]
                
                if method == 'FriendQ':
                    values, nextMove = self.selectionFunction_FriendQ(state_prime)
                elif method == 'CorrelatedQ':
                    values, nextMove = self.selectionFunction_CorrelatedQ(state_prime)
                elif method == 'FoeQ':
                    values, nextMove = self.selectionFunction_FoeQ(state_prime)
                elif method == 'Qlearning':
                    values, nextMove = self.selectionFunction_Qlearning(state_prime)
                
                if self.verbose:
                    print("before", self.QV[1][currentState])
                for thisPlayer in range(2):
                    v0 = values[thisPlayer]
                    currentVal = self.QV[thisPlayer][currentState][move[0], move[1]]
                    deltaQ = alpha * (rewards[thisPlayer] + gamma * v0 * (not done) - currentVal)
                    
                    deltaUpdate = max(deltaUpdate, abs(deltaQ))
                    self.QV[thisPlayer][currentState][move[0], move[1]] += deltaQ
                
                if self.verbose:
                    print("after", self.QV[1][currentState])
                    
                currentState = state_prime                
                move = nextMove
            
            deltaList.append(deltaUpdate)
            
            if self.verbose: print("End game", thisGame)
        
        print("Final alpha", alpha)
        return deltaList
    
    def selectionFunction_FriendQ(self, state_prime):
        # from all the Q => select a V and the optimal action to that V
        # get Value
        possibleActions = [self.possibleAction(0), self.possibleAction(1)]
        values = [0.0, 0.0]
        nextMove = [0, 0]
        
        for thisPlayer in range(2):
            possibleOutcomes = self.QV[thisPlayer][state_prime][possibleActions[0],]
            possibleOutcomes = possibleOutcomes[:, possibleActions[1]]
            values[thisPlayer] = np.max(possibleOutcomes)
        
            # get preferred actions from the original matrix
            action = np.argmax(possibleOutcomes)
            
            if thisPlayer == 1:
                # if player 1 => look for the action from the rows
                action = int(action / len(possibleActions[1])) # get the riow in the possibleOutcome matrix
            else:
                # if player 2 => look for the action from the columns
                action = action % len(possibleActions[1])
            
            nextMove[thisPlayer] = possibleActions[thisPlayer][action]
        
        return values, nextMove
    
    def selectionFunction_CorrelatedQ(self, thisState):
        possibleActions_0 = self.possibleAction(0)
        possibleActions_1 = self.possibleAction(1)
        
        # 1. create payoff matrix
        A = np.zeros((len(possibleActions_0), len(possibleActions_1), 2))
        for i in range(len(possibleActions_0)):
            for j in range(len(possibleActions_1)):
                A[i, j, 0] = self.QV[0][thisState][i, j]
                A[i, j, 1] = self.QV[1][thisState][i, j]
                
        # 2. use ce to get the Value and the equilibrium distribution
        probs = np.array(CE_Solver().ce(A)['x']).reshape((1, -1))
        
        values = [0.0, 0.0]
        for thisPlayer in range(2):
            payoff = A[:, :, thisPlayer].reshape((1, -1))
            values[thisPlayer] = np.sum(probs * payoff)
        
        probs_flatten = np.ravel(probs)
        
        if np.any(probs_flatten < 0):
            probs_flatten[probs_flatten < 0] = 0
        
        # 3. select next move based on probability
        nextMoveList = [[i, j] for i in possibleActions_0 for j in possibleActions_1]
        assert(len(nextMoveList) == len(probs_flatten))
        
        nextMove = np.random.choice(range(len(probs_flatten)), p = probs_flatten)
        nextMove = nextMoveList[nextMove]

        return values, nextMove
    
    def selectionFunction_FoeQ(self, state_prime):
        possibleActions = [self.possibleAction(0), self.possibleAction(1)]
        values = [0.0, 0.0]
        nextMove = [0, 0]
        
        for thisPlayer in range(2):
            # slice according to possible action
            payoff = self.QV[thisPlayer][state_prime][possibleActions[0],]
            payoff = payoff[:, possibleActions[1]]
            
            if thisPlayer == 1:
                payoff = payoff.T
                
            tmp = Foe_Solver().maxmin(payoff)
            values[thisPlayer] = tmp[0]
            
            probs = [max(i, 0) for i in tmp[1:]]
            nextMove_tmp = np.random.choice(range(len(possibleActions[thisPlayer])), p = probs)
            nextMove[thisPlayer] = possibleActions[thisPlayer][nextMove_tmp]
        
        return values, nextMove
    
    
    def selectionFunction_Qlearning(self):
        pass
    
    
game = SoccerGame()
game.reset()
qlearning = Qlearning(game, verbose = False)

#deltaList = qlearning.simulate(500, method = "FriendQ")
#deltaList = qlearning.simulate(500, method = "CorrelatedQ")
deltaList = qlearning.simulate(5, method = "FoeQ")

plt.plot(deltaList)

#game.reset_default()
#done = False
#for i in range(5):
#    game.printState_graph()
#    state = game.getStateID()
#    tmp, action = qlearning.selectionFunction_FoeQ(state)
#    p0, p1, done = game.nextState(action)
#    if done:
#        game.printState_graph()
#        break
        
#def test():
#    game.reset_default()
#    game.printState_graph()
#    
#    state = game.getStateID()
#    qlearning = Qlearning(game)
#    tmp, action = qlearning.selectionFunction_CorrelatedQ(state)
#
