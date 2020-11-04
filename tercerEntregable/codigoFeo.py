#------------------------------------------------------------------------------------------------------------------
#   Tic Tac Toe game.
#
#   This code is an adaptation of the Tic Tac Toe bot described in:
#   Artificial intelligence with Python. Alberto Artasanchez and Prateek Joshi. 2nd edition, 2020, 
#   editorial Pack. Chapter 13.
#
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#   Imports
#------------------------------------------------------------------------------------------------------------------

from easyAI import TwoPlayersGame, Human_Player, AI_Player, Negamax

#------------------------------------------------------------------------------------------------------------------
#   Class definitions
#------------------------------------------------------------------------------------------------------------------

class TicTacToeGameController(TwoPlayersGame):
    """ Class that is used to play the TIC TAC TOE game. """

    def __init__(self, players):
        """ 
            This constructor initializes the game according to the specified players.

            players : The list with the player objects.
        """

        # Define the players
        self.players = players

        # Define who starts the game
        self.nplayer = 1 

        # Define the board
        #self.board = [0] * 9
        self.board = [0] * 12
    
    def show(self):
        """ This method prints the current game state. """
        symbols = [' ', '-', '-' ]
        symbols2 = [' ', '|' , '|']
        print('o', symbols[self.board[0]], 'o', symbols[self.board[1]], 'o')
        print(symbols2[self.board[2]], ' ', symbols2[self.board[3]], ' ', symbols2[self.board[4]])
        print('o', symbols[self.board[5]], 'o', symbols[self.board[6]], 'o')
        print(symbols2[self.board[7]], ' ', symbols2[self.board[8]], ' ', symbols2[self.board[9]])
        print('o', symbols[self.board[10]], 'o', symbols[self.board[11]], 'o')
        

    def possible_moves(self):
        """ This method returns the possible moves according to the current game state. """        
        return [a + 1 for a, b in enumerate(self.board) if b == 0]
    
    def make_move(self, move):
        """ 
            This method executes the specified move.

            move : The move to execute.
        """
        self.board[int(move) - 1] = self.nplayer

    
    def loss_condition(self):
        """ This method returns whether the opponent has three in a line. """
        #possible_combinations = [[1,2,3], [4,5,6], [7,8,9], [1,4,7], [2,5,8], [3,6,9], [1,5,9], [3,5,7]]
        possible_combinations = [[1,3,4,6], [2,4,5,7], [6,8,9,11],[7,9,10,12]]
        return any([all([(self.board[i-1] == self.nopponent)  for i in combination]) for combination in possible_combinations]) 
    
    def is_over(self):
        """ This method returns whether the game is over. """
        return (self.possible_moves() == []) or self.loss_condition()
        
    def scoring(self):
        """ This method computes the game score (-100 for loss condition, 0 otherwise). """
        return -100 if self.loss_condition() else 0

#------------------------------------------------------------------------------------------------------------------
#   Main function
#------------------------------------------------------------------------------------------------------------------
def main():

    # Search algorithm of the AI player
    algorithm = Negamax(7)

    # Start the game
    TicTacToeGameController([Human_Player(), AI_Player(algorithm)]).play()

if __name__ == '__main__':
    main()

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------