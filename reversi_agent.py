"""
This module contains agents that play reversi.

Version 3.1
"""

import abc
import random
import asyncio
import traceback
import time
from multiprocessing import Process, Value
import sys
import numpy as np
import gym
import boardgame2 as bg2



_ENV = gym.make('Reversi-v0')
_ENV.reset()


def transition(board, player, action):
    """Return a new board if the action is valid, otherwise None."""
    if _ENV.is_valid((board, player), action):
        new_board, __ = _ENV.get_next_state((board, player), action)
        return new_board
    return None


class ReversiAgent(abc.ABC):
    """Reversi Agent."""

    def __init__(self, color):
        """
        Create an agent.
        
        Parameters
        -------------
        color : int
            BLACK is 1 and WHITE is -1. We can get these constants
            from bg2.BLACK and bg2.WHITE.

        """
        super().__init__()
        self._move = None
        self._color = color
    
    @property
    def player(self):
        """Return the color of this agent."""
        return self._color

    @property
    def pass_move(self):
        """Return move that skips the turn."""
        return np.array([-1, 0])

    @property
    def best_move(self):
        """Return move after the thinking.
        
        Returns
        ------------
        move : np.array
            The array contains an index x, y.

        """
        if self._move is not None:
            return self._move
        else:
            return self.pass_move

    async def move(self, board, valid_actions):
        """Return a move. The returned is also availabel at self._move."""
        self._move = None
        output_move_row = Value('d', -1)
        output_move_column = Value('d', 0)
        try:
            # await self.search(board, valid_actions)    
            p = Process(
                target=self.search, 
                args=(
                    self._color, board, valid_actions, 
                    output_move_row, output_move_column))
            p.start()
            while p.is_alive():
                await asyncio.sleep(0.1)
        except asyncio.CancelledError as e:
            print('The previous player is interrupted by a user or a timer.')
        except Exception as e:
            print(type(e).__name__)
            print('move() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)
        finally:
            p.kill()
            self._move = np.array(
                [output_move_row.value, output_move_column.value],
                dtype=np.int32)
        return self.best_move

    @abc.abstractmethod
    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        """
        Set the intended move to self._move.
        
        The intended move is a np.array([r, c]) where r is the row index
        and c is the column index on the board. [r, c] must be one of the
        valid_actions, otherwise the game will skip your turn.

        Parameters
        -------------------
        board : np.array
            An 8x8 array that contains 
        valid_actions : np.array
            An array of shape (n, 2) where n is the number of valid move.

        Returns
        -------------------
        None
            This method should set value for 
            `output_move_row.value` and `output_move_column.value` 
            as a way to return.

        """
        raise NotImplementedError('You will have to implement this.')


class RandomAgent(ReversiAgent):
    """An agent that move randomly."""
    
    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # If you want to "simulate a move", you can call the following function:
        # transition(board, self.player, valid_actions[0])

        # To prevent your agent to fail silently we should an
        # explicit trackback printout.
        try:
            # while True:
            #     pass
            time.sleep(3)
            randidx = random.randint(0, len(valid_actions) - 1)
            random_action = valid_actions[randidx]
            output_move_row.value = random_action[0]
            output_move_column.value = random_action[1]
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)

class SmartAgent(ReversiAgent):
    """Get Opponet function """
    
    def getOpponent(self, color):
        if color == 1:
            return -1
        else:
            return 1

    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # If you want to "simulate a move", you can call the following function:
        # transition(board, self.player, valid_actions[0])

        # To prevent your agent to fail silently we should an
        # explicit trackback printout.
        try:
            # while True:
            #     pass
            best, best_action = self.minimax(3, 0, board, valid_actions, True, - sys.maxsize -1, sys.maxsize)
            '''if best_action is None and valid_actions is not None:
                time.sleep(0.01625)
                print(" Smart AI cannot making the decision")   
                time.sleep(0.0625)    
                print(" Switch to Random Decided")
                time.sleep(0.0625)  
                randidx = random.randint(0, len(valid_actions) - 1)
                random_action = valid_actions[randidx]
                output_move_row.value = random_action[0]
                output_move_column.value = random_action[1]
                print(" Smart AI Random Selected:"+ str(random_action))
                time.sleep(0.0625)
            el'''
            if best_action is not None:
                time.sleep(0.03125)  
            # We can decided to decrease sleep time or remove it with print output
                print(" Smart AI is making the decision")
                time.sleep(0.03125)    
                output_move_row.value = best_action[0]
                output_move_column.value = best_action[1]
                print(" Smart AI Selected:" + str(best_action)+ 'because evalGives ' + str(best))
                time.sleep(0.03125)
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)
    
    def minimax(self, depth, level, board, valid_actions, maximizingPlayer , alpha , beta): 
    # Terminating condition. i.e  
    # leaf node is reached
        best_action = None
        if depth == 0:  
            return self.evalFunction(board, self.player) #Eval Function
        if maximizingPlayer:
            maxEval = - sys.maxsize - 1
            maxAlpha = alpha
            # Recur for left and right children
            for x in valid_actions:
                #new_board, new_opponentValidActions = self.createState(board, x, self.player)
                new_board = transition(board,self.player, x)
                new_opponentValidActions = self.get_validActions(new_board, self.getOpponent(self.player))
                val = self.minimax(depth - 1, level + 1, new_board, new_opponentValidActions, not maximizingPlayer, maxAlpha, beta)
                # Alpha Beta Pruning
                if maxEval < val:
                    maxEval = val
                    if level == 0:
                        best_action = x
                maxAlpha = max(maxAlpha, val)
                if beta <= maxAlpha:
                    break
            if level != 0:     
                return maxEval
            elif best_action is not None:
                return maxEval, best_action
            else:
                print('===================================Random======================================')
                return maxEval, valid_actions[random.randint(0, len(valid_actions) - 1)]
        else:
            minBeta = beta 
            minEval = sys.maxsize
            # Recur for left and
            # right children
            for x in valid_actions:
                #new_board, new_opponentValidActions = self.createState(board, x, self.getOpponent(self.player))
                new_board = transition(board, self.getOpponent(self.player), x)
                new_opponentValidActions = self.get_validActions(new_board, self.player)
                val = self.minimax(depth - 1, level + 1, new_board, new_opponentValidActions, not maximizingPlayer, alpha, minBeta)
                if minEval > val:
                    minEval = val
                    if level == 0:
                        best_action = x
                # Alpha Beta Pruning
                minBeta = min(minBeta, val)
                if minBeta <= alpha:
                    break 
            if level != 0:     
                return minEval
            elif best_action is not None:
                return minEval, best_action
            else:
                print('===================================Random======================================')
                return minEval, valid_actions[random.randint(0, len(valid_actions) - 1)]

    def get_validActions(self, board, player):
        valids = _ENV.get_valid((board, player))
        valids = np.array(list(zip(*valids.nonzero())))
        return valids

    def evalFunction(self, board, color):
        weighing_board = ([150,-70,50,20,20,50,-70,150],
                        [-70,-150,-30,-30,-30,-30,-150,-70],
                        [50,-30,50,0,0,50,-30,50],
                        [20,-30,0,0,0,0,-30,20],
                        [20,-30,0,0,0,0,-30,20],
                        [50,-30,50,0,0,50,-30,50],
                        [-70,-150,-30,-30,-30,-30,-150,-70],
                        [150,-70,50,20,20,50,-70,150])
        weighing_board = np.asarray(weighing_board)
        val = 0
        for x,y in np.nditer([weighing_board,board]):
            if y==color:
                val += x
        opponentScore = np.sum(board == self.getOpponent(color))
        myScore = np.sum(board == color)
        return val + ((myScore - opponentScore)*50)

class PoohAgent(ReversiAgent):

    def __index__(self):
        super(minimax, self)
        # self.transpositionTable = set()

    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        if self._color == 1:
            evaluation, bestAction = self.minimax(board, valid_actions, 3, 0, - sys.maxsize - 1, sys.maxsize, True)
        else:
            evaluation, bestAction = self.minimax(board, valid_actions, 3, 0, - sys.maxsize - 1, sys.maxsize, True)
        # self.createState(board, valid_actions, self._color)
        
        print("Me Selected: " + str(bestAction))
        if bestAction is not None:
            output_move_row.value = bestAction[0]
            output_move_column.value = bestAction[1]
        if bestAction is None:
            randidx = random.randint(0, len(valid_actions) - 1)
            random_action = valid_actions[randidx]
            output_move_row.value = random_action[0]
            output_move_column.value = random_action[1]

    def minimax(self, board: np.array, validActions: np.array, depth: int, levelCount: int, alpha: int, beta: int,
                maximizingPlayer: bool):
        if depth == 0:
            return self.evaluateStatistically(board, self.player)

        bestAction: np.array = None
        if maximizingPlayer:
            mAlpha: int = alpha
            maxEval: int = - sys.maxsize - 1 # -float("int")
            player: int = self._color

            for action in validActions:
                newState, newValidActions = self.createState(board, action, player)
                evaluation = self.minimax(newState, newValidActions
                                          , depth - 1, levelCount + 1, mAlpha, beta, not maximizingPlayer)

                if maxEval < evaluation:
                    maxEval = evaluation

                    if levelCount == 0:
                        bestAction = action

                mAlpha = max(mAlpha, evaluation)
                if beta <= mAlpha:
                    break
            if levelCount != 0:
                return maxEval
            else:
                return maxEval, bestAction
        else:
            mBeta: int = beta
            minEval: int = sys.maxsize
            player: int = self.getOpponent(self._color)

            for action in validActions:
                newState, newValidActions = self.createState(board, action, player)
                evaluation = self.minimax(newState, newValidActions
                                          , depth - 1, levelCount + 1, alpha, mBeta, not maximizingPlayer)

                if minEval > evaluation:
                    minEval = evaluation

                    if levelCount == 0:
                        bestAction = action

                mBeta = min(mBeta, evaluation)
                if mBeta <= alpha:
                    break
            if levelCount != 0:
                return minEval
            else:
                return minEval, bestAction

    def evaluateStatistically(self, board: np.array, color) -> int:
        opponentScore = np.sum(board == self.getOpponent(color))
        myScore = np.sum(board == color)
        #print(myScore - opponentScore)
        return myScore - opponentScore

    @staticmethod
    def getOpponent(player: int):
        if player == 1:
            return -1
        else:
            return 1

    def createState(self, board: np.array, action: np.array, player: int) -> (np.array, np.array):
        newState: np.array = transition(board, player, action)

        validMoves: np.array = _ENV.get_valid((newState, self.getOpponent(player)))
        validMoves: np.array = np.array(list(zip(*validMoves.nonzero())))

        return newState, validMoves
    
class ImprovedAgent(ReversiAgent):

    def getOp(self):
        if self.player == bg2.BLACK:
            return bg2.WHITE
        else:
            return bg2.BLACK
        
    def search(self, color, board, valid_actions, out_row, out_col):
        best_move = valid_actions[0]
        best_move_score = float('-inf')
        depth=3
        e=ImprovedAgentEvaluator(self.player)
        for i in valid_actions:
            new_board = transition(board,self.player,i)
            
            new_board_score = self.min_value(new_board,self.player,depth-1,float('-inf'),float('inf'),e)
            if best_move_score<new_board_score:
                best_move=i
                out_row.value = best_move[0]
                out_col.value = best_move[1]
                best_move_score=new_board_score

    def max_value(self,board,player,depth,a,b,e):
        
        valids = _ENV.get_valid((board,player))
        valids = np.array(list(zip(*valids.nonzero())))
        if depth==0 or valids.size<=0:
            return e.eval(board,valids)
        
        if valids.size<=0:
            return self.min_value(board,player,depth-1,a,b,e)
        
        score= float('-inf')

        for i in valids:
            new_board = transition(board,player,i)
            new_board_score = self.min_value(new_board,player,depth-1,a,b,e)
            if new_board_score>score:
                score = new_board_score
            if score>a:
                a=score
            if b<=a:
                break
        return score
        
    def min_value(self,board,player,depth,a,b,e):
        op = self.getOp()
        
        valids = _ENV.get_valid((board,op))
        valids = np.array(list(zip(*valids.nonzero())))
        if depth==0 or valids.size<=0:
            return e.eval(board,valids)
        
        if valids.size<=0:
            return self.max_value(board,player,depth-1,a,b,e)
        
        score= float('inf')

        for i in valids:
            new_board = transition(board,op,i)
            new_board_score = self.max_value(new_board,player,depth-1,a,b,e)
            if new_board_score<score:
                score = new_board_score
            if score<b:
                b=score
            if b<=a:
                break
            
        return score

class ImprovedAgentEvaluator():

    def __init__(self,player):
        self.player = player
        if self.player == bg2.BLACK:
            self.op = bg2.WHITE
        else:
            self.op = bg2.BLACK

    def eval(self,board,mymove):
        countPlace = np.count_nonzero(board)
        if countPlace<20:
            return 25*self.evalMobility(board,mymove)+400*self.evalCorner(board)+25*self.evalPosition(board)
        elif countPlace<55:
            return 30*self.evalMobility(board,mymove)+5*self.evalDiff(board)+400*self.evalCorner(board)+15*self.evalPosition(board)
        else:
            return 40*self.evalMobility(board,mymove)+20*self.evalDiff(board)+400*self.evalCorner(board)+40*self.evalPosition(board)

    def evalMobility(self,board,mymove):
        
        opMove = _ENV.get_valid((board,self.op))
        mymove = mymove.size
        opMove = np.array(list(zip(*opMove.nonzero()))).size

        return (mymove-opMove)/(mymove+opMove+1)*100
        
    def evalDiff(self,board):
        return self.player*np.sum(board)/np.count_nonzero(board)*100

    def evalCorner(self,board):
        my=0
        op=0
        if board[0][0] == self.player:my+=1
        elif board[0][0] == self.op:op+=1
        if board[0][7] == self.player:my+=1
        elif board[0][7] == self.op:op+=1
        if board[7][0] == self.player:my+=1
        elif board[7][0] == self.op:op+=1
        if board[7][7] == self.player:my+=1
        elif board[7][7] == self.op:op+=1

        return (my-op)/(my+op+1)*100
    
    def evalPosition(self,board):
        boardScore = [[150,-70,50,20,20,50,-70,150],
                    [-70,-150,-30,-30,-30,-30,-150,-70],
                    [50,-30,50,0,0,50,-30,50],
                    [20,-30,0,0,0,0,-30,20],
                    [20,-30,0,0,0,0,-30,20],
                    [50,-30,50,0,0,50,-30,50],
                    [-70,-150,-30,-30,-30,-30,-150,-70],
                    [150,-70,50,20,20,50,-70,150]]
        
        if board[0][0] !=0:
            for i in range(0,4):
                for j in range(0,4):
                    boardScore[i][j]=0
        if board[0][7] !=0:
            for i in range(0,4):
                for j in range(4,8):
                    boardScore[i][j]=0
        if board[7][0] !=0:
            for i in range(4,8):
                for j in range(0,4):
                    boardScore[i][j]=0
        if board[7][7] !=0:
            for i in range(4,8):
                for j in range(4,8):
                    boardScore[i][j]=0
        my=0
        op=0
        for i in range(0,8):
            for j in range(0,8):
                if board[i][j]==self.player:my+=boardScore[i][j]
                if board[i][j]==self.op:op+=boardScore[i][j]
        return (my-op)/(my+op+1)*100