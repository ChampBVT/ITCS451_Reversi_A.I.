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
        """ Set the intended move to the value of output_moves. """
        # If you want to "simulate a move", you can call the following function:
        # transition(board, self.player, valid_actions[0])

        # To prevent your agent to fail silently we should an
        # explicit trackback printout.
        try:
            # while True:
            #     pass
            depth = 3
            best, best_action = self.minimax(depth, 0, board, valid_actions, True, - sys.maxsize -1, sys.maxsize)
            if best_action is not None: 
                output_move_row.value = best_action[0]
                output_move_column.value = best_action[1]
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)
    
    """MinMax algorithm"""
    def minimax(self, depth, level, board, valid_actions, maximizingPlayer , alpha , beta): 
        best_action = None
        if depth == 0:  
            return self.evalFunction(board, self.player)
        if maximizingPlayer:
            maxEval = - sys.maxsize - 1
            maxAlpha = alpha
            for x in valid_actions:
                new_board = transition(board,self.player, x)
                new_opponentValidActions = self.get_validActions(new_board, self.getOpponent(self.player))
                val = self.minimax(depth - 1, level + 1, new_board, new_opponentValidActions, not maximizingPlayer, maxAlpha, beta)
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
                return self.select_weighing(valid_actions, board, maximizingPlayer)
        else:
            minBeta = beta 
            minEval = sys.maxsize
            for x in valid_actions:
                new_board = transition(board, self.getOpponent(self.player), x)
                new_opponentValidActions = self.get_validActions(new_board, self.player)
                val = self.minimax(depth - 1, level + 1, new_board, new_opponentValidActions, not maximizingPlayer, alpha, minBeta)
                if minEval > val:
                    minEval = val
                    if level == 0:
                        best_action = x
                minBeta = min(minBeta, val)
                if minBeta <= alpha:
                    break 
            if level != 0:     
                return minEval
            elif best_action is not None:
                return minEval, best_action
            else:
                return self.select_weighing(valid_actions, board, maximizingPlayer)

    """In case minmax couldn't find best action"""
    def select_weighing(self, valid_actions, board, maximizingPlayer):
        if maximizingPlayer:
            maxVal = -sys.maxsize -1
            for x in valid_actions:
                new_board = transition(board, self.player, x)
                val = self.evalFunction(new_board, self.player)
                print("max ",x,val)
                if maxVal < val:
                    maxVal = val
                    best_action = x
            return maxVal, best_action
        else:
            minVal = sys.maxsize
            for x in valid_actions:
                new_board = transition(board, self.player, x)
                val = self.evalFunction(new_board, self.player)
                print("min ",x,val)
                if minVal > val:
                    minVal = val
                    best_action = x
            return minVal, best_action

    """Get valid actions of that player"""
    def get_validActions(self, board, player):
        valids = _ENV.get_valid((board, player))
        valids = np.array(list(zip(*valids.nonzero())))
        return valids

    """Evaluation function"""
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
        return val + ((myScore - opponentScore)*40)