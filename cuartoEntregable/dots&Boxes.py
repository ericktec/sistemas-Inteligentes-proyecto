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
from Clasificador import Clasificador
import os
import time





#------------------------------------------------------------------------------------------------------------------
#   Class definitions
#------------------------------------------------------------------------------------------------------------------

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class TicTacToeGameController(TwoPlayersGame):
    """ Class that is used to play the TIC TAC TOE game. """

    def __init__(self, players):

        for i, player in enumerate(players):
            player.score = 0

        
        """ 
            This constructor initializes the game according to the specified players.

            players : The list with the player objects.
        """

        # Define the players
        self.players = players

        # Define who starts the game
        self.nplayer = 1

        # Define the board
        self.board = [0] * 24


        ##possible marks to know where the points are
        self.possiblePoints = [[1,4,5,8], [2,5,6,9], [3,6,7,10], 
                            [8, 11, 12, 15],  [9, 12, 13, 16], [9, 13, 14, 17], 
                            [15, 18, 19, 22], [16, 19, 20, 23], [17, 20, 21, 24]]

        

        #list to know whether the points are take or not, score, taken by 1 (player1) or 2 (player2) in the list, 0 for none
        self.pointsTaken = [0]*9
    
    def show(self):
        """ This method prints the current game state. """
        p1H = bcolors.OKBLUE + '-' +bcolors.ENDC
        p1V = bcolors.OKBLUE + '|' +bcolors.ENDC
        

        p2H = bcolors.OKGREEN + '-' +bcolors.ENDC
        p2V = bcolors.OKGREEN + '|' +bcolors.ENDC

        takenColors = [' ', bcolors.OKBLUE+'1'+bcolors.ENDC, bcolors.OKGREEN+"2"+bcolors.ENDC]

        horizontal = [' ',p1H, p2H]
        vertical = [' ', p1V, p2V]



        ##Clear command for windows
        #os.system('cls')

        ##clear command for linux & mac:
        # os.system('clear')

        
        print(bcolors.BOLD+'o '+bcolors.ENDC,horizontal[self.board[0]], bcolors.BOLD+' o '+bcolors.ENDC,horizontal[self.board[1]], bcolors.BOLD+' o '+bcolors.ENDC,horizontal[self.board[2]], bcolors.BOLD+' o '+bcolors.ENDC)
        print(vertical[self.board[3]], takenColors[self.pointsTaken[0]],' ',vertical[self.board[4]], takenColors[self.pointsTaken[1]],' ',vertical[self.board[5]], takenColors[self.pointsTaken[2]],' ', vertical[self.board[6]])
        print(bcolors.BOLD+'o '+bcolors.ENDC,horizontal[self.board[7]], bcolors.BOLD+' o '+bcolors.ENDC,horizontal[self.board[8]], bcolors.BOLD+' o '+bcolors.ENDC,horizontal[self.board[9]], bcolors.BOLD+' o '+bcolors.ENDC)
        print(vertical[self.board[10]], takenColors[self.pointsTaken[3]],' ',vertical[self.board[11]], takenColors[self.pointsTaken[4]],' ',vertical[self.board[12]], takenColors[self.pointsTaken[5]],' ', vertical[self.board[13]])
        print(bcolors.BOLD+'o '+bcolors.ENDC,horizontal[self.board[14]], bcolors.BOLD+' o '+bcolors.ENDC,horizontal[self.board[15]], bcolors.BOLD+' o '+bcolors.ENDC,horizontal[self.board[16]], bcolors.BOLD+' o '+bcolors.ENDC)
        print(vertical[self.board[17]], takenColors[self.pointsTaken[6]],' ',vertical[self.board[18]], takenColors[self.pointsTaken[7]],' ',vertical[self.board[19]], takenColors[self.pointsTaken[8]],' ', vertical[self.board[20]])
        print(bcolors.BOLD+'o '+bcolors.ENDC,horizontal[self.board[21]], bcolors.BOLD+' o '+bcolors.ENDC,horizontal[self.board[22]], bcolors.BOLD+' o '+bcolors.ENDC,horizontal[self.board[23]], bcolors.BOLD+' o '+bcolors.ENDC)
        print()
        [self.p1, self.p2] = self.printScore()
        

    def possible_moves(self):
        """ This method returns the possible moves according to the current game state. """      
        return [a + 1 for a, b in enumerate(self.board) if b == 0]
    
    def make_move(self, move):
        """ 
            This method executes the specified move.

            move : The move to execute.
        """
        self.board[int(move) - 1] = self.nplayer
        self.updatePoints(self.nplayer)



    
    
    
    def is_over(self):
        """ This method returns whether the game is over. """
        return self.possible_moves() == [] 
    
    def loss_condition(self):
        return 
        
    def scoring(self):
        """ This method computes the game score (-100 for loss condition, 0 otherwise). """
        p1 = 0
        p2 = 0
        for n in self.pointsTaken:
            if n==1:
                p1+=1
            elif n==2:
                p2+=1

        if self.nplayer == 1:

            if p1 < p2:
                return -100
            elif p1 == p2:
                return -100
            else:
                return 0
        else:

            if p2 > p1:
                return 0
            elif p1 == p2:
                return -100
            else:
                return -100
        

    def updatePoints(self, playerN):
        currentPoints = [all([(self.board[i-1] != 0) for i in combination]) for combination in self.possiblePoints]
        i=0
        for possiblePoints in currentPoints:
            if possiblePoints and self.pointsTaken[i]==0:
                self.pointsTaken[i]=playerN
                self.nplayer = self.nopponent
            i+=1
        
        return

    def isOver(self):
        if self.possible_moves() == []:
            return True
        else:
            return False

    def printScore(self):
        p1 = 0
        p2 = 0
        for n in self.pointsTaken:
            if n==1:
                p1+=1
            elif n==2:
                p2+=1
        print(bcolors.WARNING+'SCORE: '+bcolors.ENDC, bcolors.OKBLUE + 'P1: '+bcolors.ENDC, bcolors.WARNING+'',p1,'  - '+bcolors.ENDC, bcolors.OKGREEN+' P2: '+bcolors.ENDC,bcolors.WARNING+'',p2,'',bcolors.ENDC)
        
        return p1, p2

    def lose(self):
        return self.opponent.score > self.score

    


#------------------------------------------------------------------------------------------------------------------
#   Main function
#------------------------------------------------------------------------------------------------------------------
def main():


    clasificador = Clasificador()

    # Search algorithm of the AI player
    algorithm = Negamax(5)

    # Start the game
    TicTacToeGameController([Human_Player(clasificador), AI_Player(algorithm)]).play()
    #TicTacToeGameController([Human_Player(), Human_Player()]).play()



if __name__ == '__main__':
    main()


#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------



# NEW BOARD
# o - o - o - o
# |   |   |   |
# o - o - o - o
# |   |   |   |
# o - o - o - o
# |   |   |   |
# o - o - o - o