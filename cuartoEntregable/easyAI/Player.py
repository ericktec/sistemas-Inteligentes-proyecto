"""
This module implements the Player (Human or AI), which is basically an
object with an ``ask_move(game)`` method
"""
try:
    input = raw_input
except NameError:
    pass


class Human_Player:
    """
    Class for a human player, which gets asked by text what moves
    she wants to play. She can type ``show moves`` to display a list of
    moves, or ``quit`` to quit the game.
    """

    def __init__(self, clasificador, name = 'Human' ):
        self.name = name
        self.clasificador = clasificador

    def ask_move(self, game):

        # 101 = Flexionando a la izquierda, 102 = Flexionando a la derecha, 103  = Mano cerrada
        initial_option = 0
        possible_moves = game.possible_moves()
        gettingInput = True
        # The str version of every move for comparison with the user input:
        possible_moves_str = list(map(str, game.possible_moves()))
        move = "NO_MOVE_DECIDED_YET"
        while True:
            while gettingInput:
                print("La señal obtenida es : ", self.clasificador.getInput()[0])
                option = int(self.clasificador.getInput()[0].item())
                print("Current move is ", initial_option)
                if(option == 101 and initial_option>0):
                    initial_option-=1
                elif(option == 102 and initial_option<24):
                    initial_option+=1
                elif(option == 103 and initial_option<24):
                    move = str(initial_option)
                    gettingInput = False
            

            if move == 'show moves':
                print ("Possible moves:\n"+ "\n".join(
                       ["#%d: %s"%(i+1,m) for i,m in enumerate(possible_moves)])
                       +"\nType a move or type 'move #move_number' to play.")

            elif move == 'quit':
                raise KeyboardInterrupt

            elif move.startswith("move #"):
                # Fetch the corresponding move and return.
                move = possible_moves[int(move[6:])-1]
                return move

            elif str(move) in possible_moves_str:
                # Transform the move into its real type (integer, etc. and return).
                move = possible_moves[possible_moves_str.index(str(move))]
                print("move is ",type(move))
                return move
            else:
                gettingInput = True

class AI_Player:
    """
    Class for an AI player. This class must be initialized with an
    AI algortihm, like ``AI_Player( Negamax(9) )``
    """

    def __init__(self, AI_algo, name = 'AI'):
        self.AI_algo = AI_algo
        self.name = name
        self.move = {}

    def ask_move(self, game):
        return self.AI_algo(game)
