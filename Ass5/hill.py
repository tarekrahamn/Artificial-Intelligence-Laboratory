import random
import numpy as np

class Queen:
    def __init__(self, N):
        self.N = N
        self.queen_loc = dict()
        self.initialize = False
        self.chess_board = [[0] * self.N for _ in range(self.N)]

    def add_queen(self):
        if not self.initialize:
            number_Q = 0
            while True:
                flag = 0
                r = np.random.randint(self.N)
                c = np.random.randint(self.N)
                for q in self.queen_loc:
                    row, col = self.queen_loc[q]
                    if (r == row and c == col) or (c == col):
                        flag = 1
                if flag == 0:
                    Q = f"Q{number_Q}"
                    if Q not in self.queen_loc:
                        self.queen_loc[Q] = []
                    self.queen_loc[Q].append(r)
                    self.queen_loc[Q].append(c)
                    self.chess_board[r][c] = Q
                    number_Q += 1
                if number_Q == self.N:
                    break
            self.initialize = True

    def get_neighbor(self, row, col):
        neighbor = []
        if 0 <= row - 1 < self.N and self.chess_board[row - 1][col] == 0:
            neighbor.append([row - 1, col])
        if 0 <= row + 1 < self.N and self.chess_board[row + 1][col] == 0:
            neighbor.append([row + 1, col])
        return neighbor

    def print_Queen(self):
        print(self.chess_board)
        for Q in self.queen_loc:
            print(f'{Q}->{self.queen_loc[Q]}')


def conflict(r1, c1, r2, c2):
    if r1 == r2:
        return True
    if c1 == c2:
        return True
    if r1 + c1 == r2 + c2:
        return True
    if r1 - c1 == r2 - c2:
        return True
    return False


def get_conflict(Q, state):
    count = 0
    for q in state:
        if q is not Q:
            r1, c1 = state[Q]
            r2, c2 = state[q]
            if conflict(r1, c1, r2, c2):
                count += 1
    return count


def calc_cost(state):
    cost = 0
    max_conflict = -999
    maxQ = None
    for Q in state:
        q_conflict = get_conflict(Q, state)
        cost += q_conflict
        if q_conflict > max_conflict:
            max_conflict = q_conflict
            maxQ = Q
    return cost // 2, max_conflict, maxQ

def generate_neighbors(queen, state):
    neighbors = []
    for Q in state:
        current_row, current_col = state[Q]

        for neighbor_row in range(queen.N):
            if neighbor_row != current_row:
                neighbor_state = state.copy()
                neighbor_state[Q] = [neighbor_row, current_col]
                neighbors.append(neighbor_state)

    return neighbors

def get_best_neighbor(queen, neighbors):
    best_neighbor = None
    best_cost = float('inf')

    for neighbor in neighbors:
        cost, _, _ = calc_cost(neighbor)

        if cost < best_cost:
            best_cost = cost
            best_neighbor = neighbor

    return best_neighbor


def hill_climbing_search(queen):
    queen.add_queen()  # Add initial state
    queen.print_Queen()
    total_cost=0
    current_state = queen.queen_loc.copy()
    current_cost, _, _ = calc_cost(current_state)

    total_cost+=current_cost
    while not goal_test(current_state):
        neighbor_states = generate_neighbors(queen, current_state)
        next_state = get_best_neighbor(queen, neighbor_states)
        next_cost, _, _ = calc_cost(next_state)

        if next_cost >= current_cost:
            break  # Local minimum reached

        current_state = next_state
        current_cost = next_cost
        total_cost +=current_cost

    return current_state,total_cost

def goal_test(state):
    cost, _, _ = calc_cost(state)
    return cost == 0

queen = Queen(4)
queen.print_Queen()
result,total_cost = hill_climbing_search(queen)
queen.queen_loc = result
queen.print_Queen()
print("Total Cost : ",total_cost)

