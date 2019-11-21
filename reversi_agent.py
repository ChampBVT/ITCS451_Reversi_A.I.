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

    def minimax(self, depth, level, board, valid_actions, maximizingPlayer, alpha, beta):  
    # Terminating condition. i.e  
    # leaf node is reached
        MAX, MIN = 1000000, -1000000
        minEval = MAX
        maxEval = MIN
        best_action = None
        if depth == 0:  
            return self.evalFunction(board, self.player) #Eval Function
        if maximizingPlayer:
            best = MIN
            # Recur for left and right children
            for x in valid_actions:
                new_board = transition(board,self.player, x)
                new_opponentValidActions = self.get_validActions(new_board, self.getOpponent(self.player))
                val = self.minimax(depth - 1, level + 1, new_board, new_opponentValidActions, False, alpha, beta)
                best = max(best, val)
                alpha = max(alpha, best)
                print(" ", x, val)
                # Alpha Beta Pruning
                if maxEval < best:
                    maxEval = best
                    if level == 0:
                        best_action = x
                if beta <= alpha:
                    print('kuyyyyyyyyyyyyyyyyyyyyyyyy')
                    break
            if(level != 0):     
                return best
            else:
                return best, best_action
        else: 
            best = MAX 
            # Recur for left and
            # right children
            for x in valid_actions:
                new_board = transition(board, self.getOpponent(self.player), x)
                new_opponentValidActions = self.get_validActions(new_board, self.player)
                val = self.minimax(depth - 1, level + 1, new_board, new_opponentValidActions, True, alpha, beta)
                best = min(best, val)
                beta = min(beta, best)
                print(" ", x, val)
                if minEval > best:
                    minEval = best
                    if level == 0:
                        best_action = x
                # Alpha Beta Pruning  
                if beta <= alpha:
                    print('kuyyyyyyyyyyyyyyyyyyyyyyyy')
                    break 
            if(level != 0):     
                return best
            else:
                return best, best_action

    def get_validActions(self, board, player):
        valids = _ENV.get_valid((board, player))
        valids = np.array(list(zip(*valids.nonzero())))
        return valids

    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # If you want to "simulate a move", you can call the following function:
        # transition(board, self.player, valid_actions[0])

        # To prevent your agent to fail silently we should an
        # explicit trackback printout.
        try:
            # while True:
            #     pass
            best, best_action = self.minimax(10, 0, board, valid_actions, True, 100000, -100000)
            if best_action is None and valid_actions is not None:
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
            elif best_action is not None:
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

    def evalFunction(self, board, color):
        opponentScore = np.sum(board == self.getOpponent(color))
        myScore = np.sum(board == color)
        return myScore - opponentScore