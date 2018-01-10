"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import numpy
import math

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


    

def custom_score(game, player):
    playerLocation = game.get_player_location(player)
    opponentLocation = game.get_player_location(game.get_opponent(player))
    
    my_moves = float(len(game.get_legal_moves(player)))
    my_opponent_moves = float(len(game.get_legal_moves(game.get_opponent(player))))
    

    return math.sqrt((playerLocation[0]-opponentLocation[0]) * (playerLocation[0]-opponentLocation[0]) + (playerLocation[1]-opponentLocation[1]) * (playerLocation[1]-opponentLocation[1])) * (my_moves - my_opponent_moves)


def custom_score_2(game, player):
    my_moves = float(len(game.get_legal_moves(player)))
    my_opponent_moves = float(len(game.get_legal_moves(game.get_opponent(player))))
    return my_moves - 2*my_opponent_moves
    


def custom_score_3(game, player):
    my_moves = float(len(game.get_legal_moves(player)))
    my_opponent_moves = float(len(game.get_legal_moves(game.get_opponent(player))))
    
    centerPoint = (game.width/2,game.height/2)
    
    count = 0
    for move in game.get_legal_moves(player):
        result = numpy.subtract(move, centerPoint)
        result[0] = abs(result[0])
        result[1] = abs(result[1])
        
        count = count + 1 / (result[0] + result[1])
    
    for move in game.get_legal_moves(game.get_opponent(player)):
        result = numpy.subtract(move, centerPoint)
        result[0] = abs(result[0])
        result[1] = abs(result[1])
        
        count = count - (result[0] + result[1])
    return my_moves - my_opponent_moves + count


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        if best_move == (-1,-1):
            if len(game.get_legal_moves())>0:
                return game.get_legal_moves()[0]
        return best_move
    
    
    """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
    def minVal(self, game,depth,currentDepth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if currentDepth == depth:
            return self.score(game,self) 
        
        currentDepth = currentDepth + 1
        
        v = float("inf")
        
        for move in game.get_legal_moves():
            new_game = game.forecast_move(move)
            v = min(v,self.maxVal(new_game,depth,currentDepth))
        return v
    
    def maxVal(self, game,depth,currentDepth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if currentDepth == depth:
            return self.score(game,self)
        
        currentDepth = currentDepth + 1
        v = float("-inf")
        
        for move in game.get_legal_moves():
            new_game = game.forecast_move(move)
            v = max(v,self.minVal(new_game,depth,currentDepth))
        return v
       
    
    def minimax(self, game, depth):       
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()        
                
        bestMove = None
        
        bestValue = -100
        for move in game.get_legal_moves():
            new_game = game.forecast_move(move)
    
            currentValue = self.minVal(new_game,depth,1)
            if currentValue > bestValue:
                bestValue = currentValue
                bestMove = move
    
        if bestMove == None:
            if len(game.get_legal_moves()) > 0:
                return game.get_legal_moves()[0]
        return bestMove

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        self.time_left = time_left
        
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.interativeSearch(game,time_left)
            """return self.alphabeta(game, self.search_depth,alpha=float("-inf"), beta=float("inf"))"""

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        if best_move == (-1,-1):
            if len(game.get_legal_moves())>0:
                return game.get_legal_moves()[0]
        return best_move

    def minVal(self, game,depth,alpha,beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        if len(game.get_legal_moves())==0:
            return self.score(game,self) 
        
        if depth == 0:
            return self.score(game,self) 
 
        v = float("inf")
        
        for move in game.get_legal_moves():
            new_game = game.forecast_move(move)
            v = min(v,self.maxVal(new_game,depth - 1,alpha,beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    
    def maxVal(self, game,depth,alpha,beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        if len(game.get_legal_moves())==0:
            return self.score(game,self) 
        
        if depth == 0:
            return self.score(game,self) 
        
        v = float("-inf")
        
        for move in game.get_legal_moves():
            new_game = game.forecast_move(move)
            v = max(v,self.minVal(new_game,depth - 1,alpha,beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
    
    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        bestScore = -100
        bestMove = (-1,-1)
        for move in game.get_legal_moves():
            new_game = game.forecast_move(move)
            v = self.minVal(new_game,depth - 1,alpha,beta)
            if v > bestScore:
                bestScore = v
                bestMove = move
            alpha = max(alpha, v)
        return bestMove
    
    def interativeSearch(self,game,time_left):
        depth = 1
        bestMove = (-1,-1)
        try:
            while True:
                bestMove = self.alphabeta(game,depth,alpha=float("-inf"), beta=float("inf"))
                depth = depth + 1
        except SearchTimeout:
            if bestMove == (-1,-1):
                if len(game.get_legal_moves())>0:
                    return game.get_legal_moves()[0]
            return bestMove
        
    
    
    
        
