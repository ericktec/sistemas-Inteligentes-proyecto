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
        self.board = [0] * 9
    
    def show(self):
        """ This method prints the current game state. """
        print('\n'+'\n'.join([' '.join([['.', 'O', 'X'][self.board[3*j + i]]
                for i in range(3)]) for j in range(3)]))

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
        possible_combinations = [[1,2,3], [4,5,6], [7,8,9],
            [1,4,7], [2,5,8], [3,6,9], [1,5,9], [3,5,7]]

        return any([all([(self.board[i-1] == self.nopponent)
                for i in combination]) for combination in possible_combinations]) 
    
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


#board = [0]*12
#possible_combinations = [ [1,3,4,6], [2,4,5,7], [6,8,9,11],[7,9,10,12] ]

#[1,3,4,6]

# o - o   o      
# | 1 |             
# o - o   o       
#       2            
# o   o   o  


#[2,4,5,7]

# o   o - o      
#     |   |         
# o   o - o       
#                    
# o   o   o   



#[6,8,9,11]

# o   o   o      
#                   
# o - o   o       
# |   |              
# o - o   o   



#[7,9,10,12]
# o   o   o      
#                   
# o   o - o       
#     |   |          
# o   o - o   