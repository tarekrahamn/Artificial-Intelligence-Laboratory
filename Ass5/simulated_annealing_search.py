import random
import numpy as np
import math

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

def get_random_neighbor(queen, state):
    Q = random.choice(list(state.keys()))
    current_row, current_col = state[Q]
    neighbor_row = random.randint(0, queen.N - 1)
    neighbor_state = state.copy()
    neighbor_state[Q] = [neighbor_row, current_col]
    return neighbor_state

def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost < old_cost:
        return 1.0
    return math.exp((old_cost - new_cost) / temperature)

def simulated_annealing_search(queen, initial_temperature, cooling_rate):
    queen.add_queen()  # Add initial state
    queen.print_Queen()
    total_cost = 0
    current_state = queen.queen_loc.copy()
    current_cost, _, _ = calc_cost(current_state)

    total_cost += current_cost

    temperature = initial_temperature
    while temperature > 0.1 and not goal_test(current_state):
        neighbor_state = get_random_neighbor(queen, current_state)
        new_cost, _, _ = calc_cost(neighbor_state)

        if new_cost < current_cost or random.uniform(0, 1) < acceptance_probability(current_cost, new_cost, temperature):
            current_state = neighbor_state
            current_cost = new_cost

        temperature *= cooling_rate
        total_cost += current_cost

    return current_state, total_cost

def goal_test(state):
    cost, _, _ = calc_cost(state)
    return cost == 0

queen = Queen(4)
queen.print_Queen()
result, total_cost = simulated_annealing_search(queen, initial_temperature=1000, cooling_rate=0.99)
queen.queen_loc = result
queen.print_Queen()
print("Total Cost : ", total_cost)
