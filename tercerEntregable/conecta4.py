import numpy as np

from easyAI import TwoPlayersGame, Human_Player, AI_Player, Negamax 


class ConnectFour(TwoPlayersGame):


    def __init__(self, players):
        self.players = players
        self.board = np.array([[0 for i in range(7)] for j in range(6)])
        self.nplayer = 1

    def possible_moves(self):
        return [i for i in range(7) if (self.board[:, i].min() == 0)]

    def make_move(self, column):
        line = np.argmin(self.board[:, column] != 0)
        self.board[line, column] = self.nplayer

    def show(self):
        print('\n' + '\n'.join(
                        ['0 1 2 3 4 5 6', 13 * '-'] +
                        [' '.join([['.', 'O', 'X'][self.board[5 - j][i]]
                        for i in range(7)]) for j in range(6)]))

    def lose(self):
        return winner(self.board, self.nopponent)

    def is_over(self):
        return (self.board.min() > 0) or self.lose()

    def scoring(self):
        return -100 if self.lose() else 0


def winner(board, nplayer):
    
    for pos, direction in Combinations:
        streak = 0
        while (0 <= pos[0] <= 5) and (0 <= pos[1] <= 6):
            if board[pos[0], pos[1]] == nplayer:
                streak += 1
                if streak == 4:
                    return True
            else:
                streak = 0
            pos = pos + direction
    return False


Combinations = np.array([[[i, 0], [0, 1]] for i in range(6)] +
                   [[[0, i], [1, 0]] for i in range(7)] +
                   [[[i, 0], [1, 1]] for i in range(1, 3)] +
                   [[[0, i], [1, 1]] for i in range(4)] +
                   [[[i, 6], [1, -1]] for i in range(1, 3)] +
                   [[[0, i], [1, -1]] for i in range(3, 7)])

def main():
    algorithm = Negamax(7)
    game = ConnectFour([AI_Player(algorithm), Human_Player()])
    game.play()
    if game.lose():
        print("Jugador %d gano." % (game.nopponent))
    else:
        print("Empate.")


if __name__ == '__main__':
    main()