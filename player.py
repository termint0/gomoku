from random import randint
import numpy as np
from time import time

class Board:
    SIZE = 15
    WIN_CONDITION = 5
    PATTERNS = [
        ("-1 -1 -1 -1 -1", -100000000),
        ("1 1 1 1 1", 100000000),
        (" 0 -1 -1 -1 -1  0", -100000),
        ("0 1 1 1 1 0", 100000),
        (" 0 -1 -1 -1 -1  1", -100),
        (" 0  1  1  1  1 -1", 100),
        (" 1 -1 -1 -1 -1  0", -100),
        ("-1  1  1  1  1  0", 100),
        (" 0 -1 -1 -1  0", -5),
        ("0 1 1 1 0", 5),
        (" 0 -1 -1  0", -1),
        ("0 1 1 0", 1)
    ]
    '''PATTERNS_NP = np.array([
        np.ones(WIN_CONDITION, dtype = int),
        np.negative(np.ones(WIN_CONDITION, dtype = int)),
        np.array([0, 1, 1, 1, 1, 0]),
        np.array([0, -1, -1, -1, -1, 0]),
        np.array([0, 1, 1, 1, 1, -1]),
        np.array([0, -1, -1, -1, -1, 1]),
        np.array([1, -1, -1, -1, -1, 0]),
        np.array([-1, 1, 1, 1, 1, 0]),
        np.array([0, 1, 1, 1, 0]),
        np.array([0, -1, -1, -1, 0]),
        np.array([0, 1, 1, 0]),
        np.array([0, -1, -1, 0])

    ], dtype = object)
    PATTERNS_VL = {
    "[-1 -1 -1 -1 -1]" : -100000000,
    "[1 1 1 1 1]" : 100000000,
    "[ 0 -1 -1 -1 -1  0]" : -100000,
    "[0 1 1 1 1 0]" : 100000,
    "[ 0 -1 -1 -1 -1  1]" : -100,
    "[ 0  1  1  1  1 -1]" : 100,
    "[ 1 -1 -1 -1 -1  0]" : -100,
    "[-1  1  1  1  1  0]" : 100,
    "[ 0 -1 -1 -1  0]" : -5,
    "[0 1 1 1 0]" : 5,
    "[ 0 -1 -1  0]" : -1,
    "[0 1 1 0]" : 1
    
    }
'''
    def __init__(self):
        self.board = self.generate_board()

    def generate_board(self):
        board = np.zeros((self.SIZE, self.SIZE), dtype = int)
        return board

    def log_move(self, row, col, player):
        self.board[row][col] = player

    def evaluate_board(self):
        value = 0
        shrunk_board = self.shrink_board()
        size = len(shrunk_board)
        for axis in [0, 1]:
            value += np.apply_along_axis(self.evaluate_line, axis, shrunk_board).sum()
        for offset in range(size -1, - size, -1):
            value_main = self.evaluate_line(np.diagonal(shrunk_board, offset = offset))
            value_anti = self.evaluate_line(np.flip(shrunk_board, 0).diagonal(offset = offset))
            value += (value_main + value_anti)
        return value

    def shrink_board(self):
        used_squares = np.argwhere(self.board != 0)
        lowest_row, lowest_col = np.min(used_squares, axis = 0)
        highest_row, highest_col = np.max(used_squares, axis = 0)
        shrunk_board = self.board[max(lowest_row - 1, 0): min(highest_row+2, 15), max(lowest_col-1, 0): min(highest_col+2, 15)]
        return shrunk_board
        
    def evaluate_line(self, line):
        if len(line) < 4: return 0
        value = 0
        string_line = np.array_str(line)
        for pattern in self.PATTERNS:
            patt, score = pattern
            if patt in string_line:
                value += score
        return value

    def get(self, row, col):
        return self.board[row][col]


class Player:
    MAX_DEPTH = 2
    VECTORS = np.array([
        [0, 1],
        [0, -1],
        [1, 0],
        [1, 1],
        [1, -1],
        [-1, 0],
        [-1, 1],
        [-1, -1]
    ])
    def __init__(self, player_sign):
        self.sign = player_sign
        self.name = 'ViktorínŠachl'
        self.board = Board()
        self.turn = 0
        
    def get_viable_moves(self):
        used_squares = np.argwhere(self.board.board != 0).reshape((-1, 2 ))
        moves = self.get_neighbors(used_squares)
        moves = np.asarray(moves).reshape((-1, 2))
        index = (moves[:, None] != used_squares).any(-1).all(-1)
        viable_moves = moves[index]
        np.random.shuffle(viable_moves)
        return viable_moves

    def get_neighbors(self, used_squares):
        neighbors = self.VECTORS + used_squares[:, None]
        neighbors = np.clip(neighbors, 0, 14).reshape(-1, 2)
        return neighbors

    def get_empty_square(self):
        empty_squares = np.argwhere(self.board.board == 0)
        i = randint(0, len(empty_squares))
        return empty_squares[i][0], empty_squares[i][1]

    def play(self, opponent_move):
        if opponent_move == None:
            move = self.get_empty_square()
            row, col = move
            self.board.log_move(row, col, 1)
            return (row, col)
        row, col = opponent_move
        self.board.log_move(row, col, -1)
        best_move = self.start_alpha_beta()
        row, col = best_move
        self.board.log_move(row, col, 1)
        print(self.board.board)
        return (row, col)

    def start_alpha_beta(self):
        best_score = -1000000000000
        best_move = None
        for new_move in self.get_viable_moves():
            self.board.log_move(new_move[0], new_move[1], 1)
            self.viable_moves = self.get_viable_moves()
            score = self.alpha_beta(-10000000000000, 100000000000, is_max = False)
            if score > best_score: best_move = new_move; best_score = score
            print(score, new_move)
            self.board.log_move(new_move[0], new_move[1], 0)
            self.viable_moves = self.get_viable_moves()
        if best_move is None:
            print("Halp")
            best_move = self.get_empty_square()
        return best_move

    def alpha_beta(self, alpha, beta, depth = 0, is_max = True):
        board_value = self.board.evaluate_board()
        if depth == self.MAX_DEPTH or board_value > 5000000 or board_value < -5000000: return board_value
        if is_max:
            for new_move in self.get_viable_moves():
                self.board.log_move(new_move[0], new_move[1], 1)
                self.viable_moves = self.get_viable_moves()
                best_score = -10000000000000000
                score = self.alpha_beta(alpha, beta, depth + 1, is_max = False)
                best_score = max(best_score, score); alpha = max(alpha, score)
                self.board.log_move(new_move[0], new_move[1], 0)
                if beta <= alpha: break
            return best_score
        else:
            for new_move in self.get_viable_moves():
                self.board.log_move(new_move[0], new_move[1], -1)
                self.viable_moves = self.get_viable_moves()
                best_score = 1000000000000000
                score = self.alpha_beta(alpha, beta, depth + 1)
                best_score = min(best_score, score); beta = min(beta, score)
                self.board.log_move(new_move[0], new_move[1], 0)
                if beta <= alpha: break
            return best_score


        #najít alfa beta
        #opponent_move == None -> začíná/opponent špatnej move
        #randomvalid - ukázka
        #numpy?
#COCKFUCK COCKANDBALLSBALLSNASKNLSJECKJLELJ I LOVE MY (PROKOP'S) ASS <33333 
'''WHERE ARE MY BALLS'''
#HELLO WORLD SUCK MY COCK WORLD
'''I AM A DOGBOY'''
#I HAVE 1838203308203%29830010298192018191 IQ MORE THHAN THIS AI
'''(THIS ai got -7iq)'''
