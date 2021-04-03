import pandas as pd
import numpy as np
import random

class worker:
    def __init__(self, i,j):
        self.i = i
        self.j = j
        self.c = 0
        self.moves = 0

    def calculate_worker_travel_cost(self, new_i, new_j):
        manhattan_dist = (self.i - new_i)**2 + (self.j - new_j)**2
        return np.exp(manhattan_dist)

    def calculate_consumption(self, loc):
        c = (loc.m - self.calculate_worker_travel_cost(loc.i, loc.j)) / loc.p
        return c

    def search_work(self, loc):
        candidate_locations = loc.neighbors
        location_consumption = [self.calculate_consumption(neighbor) for neighbor in candidate_locations]
        max_c = max(location_consumption)

        max_loc_index = location_consumption.index(max_c)
        max_loc = candidate_locations[max_loc_index]

        return max_c, max_loc

    def update_consumer_loc(self, loc):
        max_c, max_loc = self.search_work(loc)
        self.i = max_loc.i
        self.j = max_loc.j
        self.c = max_c
        self.moves += 1



class firm:
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.profit = 0
        self.moves = 0

    def calculate_firm_travel_cost(self, new_i, new_j):
        manhattan_dist = (self.i - new_i) ** 2 + (self.j - new_j) ** 2
        return np.exp(2*manhattan_dist)

    def find_num_neighboring_firms(self, loc):
        neighbor_cities = loc.neighbors
        num_firms = sum([len(neighbor.firm_pop) for neighbor in neighbor_cities])
        return num_firms


    def calculate_profit(self, loc, num_workers):

        travel_cost = self.calculate_firm_travel_cost(loc.i, loc.j)
        y_choices = np.array(range(0, num_workers+1))
        candidate_profit = loc.p * (len(loc.firm_pop) ** loc.alpha) \
                           * (len(loc.cons_pop) ** loc.beta) * y_choices \
                           - loc.w * y_choices - travel_cost

        final_profit = max(candidate_profit)

        return final_profit

    def search_market(self, loc):
        candidate_locations = loc.neighbors
        location_profits = [self.calculate_profit(neighbor, len(neighbor.cons_pop)) for neighbor in candidate_locations]
        max_profit = max(location_profits)

        max_loc_index = location_profits.index(max_profit)
        max_loc = candidate_locations[max_loc_index]

        return max_profit, max_loc

    def update_firm_loc(self, loc):
        max_profit, max_loc = self.search_market(loc)
        self.i = max_loc.i
        self.j = max_loc.j
        self.profit = max_profit
        self.moves += 1

class location:
    def __init__(self, i,j, alpha, beta):
        self.i = i
        self.j = j
        self.alpha = alpha
        self.beta = beta

        self.neighbors = []

        self.cons_pop = []
        self.firm_pop = []

        self.p = 0
        self.w = 0
        self.m = 0

    def find_neighbors(self, grid, radius):
        rl = []
        for i in range(self.i - radius, self.i + 1 + radius):
            for j in range(self.j - radius, self.j + 1 + radius):
                if (i >= 0 and i < len(grid) and j >= 0 and j < len(grid[0])):
                    rl.append(grid[i][j])
        return rl

    def set_neighbors(self, grid, radius):
        self.neighbors = self.find_neighbors(grid, radius)

    def add_worker(self, worker):
        self.cons_pop.append(worker)

    def pop_worker(self, worker):
        worker_index = self.cons_pop.index(worker)
        return self.cons_pop.pop(worker_index)

    def add_firm(self, firm):
        self.firm_pop.append(firm)

    def pop_firm(self, firm):
        firm_index = self.firm_pop.index(firm)
        return self.firm_pop.pop(firm_index)

    def update_w_m_p(self):
        self.w = len(self.firm_pop)
        self.m = self.w * 0.9
        if len(self.cons_pop) > 5:
            self.p = random.randint(2,3)
        else:
            self.p = random.randint(1,2)

    def summarize(self, verbose):
        num_workers = len(self.cons_pop)
        num_firms = len(self.firm_pop)

        total_c = sum([worker_i.c for worker_i in self.cons_pop])
        total_profit = sum([firm_i.profit for firm_i in self.firm_pop])

        if verbose:
            return num_workers, num_firms, total_c, total_profit

        else:
            return num_workers, num_firms


    def __repr__(self):
        return f'{self.summarize(verbose=False)}'


def load_grid(nrows, ncols, radius, alpha, beta):
    grid = np.zeros(shape=(nrows, ncols))
    grid = grid.tolist()
    random.seed(1234)
    for row_num, row in enumerate(grid):
        for col_num, col in enumerate(row):
            grid[row_num][col_num] = location(row_num, col_num, alpha, beta)

            n_firms = random.randint(1,2)
            n_workers = random.randint(1,2)
            for firm_i in range(0, n_firms):
                new_firm = firm(row_num, col_num)
                grid[row_num][col_num].add_firm(new_firm)

            for worker_i in range(0, n_workers):
                new_worker = worker(row_num, col_num)
                grid[row_num][col_num].add_worker(new_worker)

    for r_i, r in enumerate(grid):
        for c_j, c in enumerate(r):
            grid[r_i][c_j].set_neighbors(grid, radius)
            grid[r_i][c_j].update_w_m_p()

    return grid

def simulate(years, grid):
    for year in range(0, years):
        for r, row in enumerate(grid):
            for c, col in enumerate(row):

                for worker_i in grid[r][c].cons_pop:
                    worker_i.update_consumer_loc(grid[r][c])

                    if worker_i not in worker_i.search_work(grid[r][c])[1].cons_pop:
                        grid[r][c].pop_worker(worker_i)
                        worker_i.search_work(grid[r][c])[1].add_worker(worker_i)

                for firm_i in grid[r][c].firm_pop:
                    firm_i.update_firm_loc(grid[r][c])

                    if firm_i not in firm_i.search_market(grid[r][c])[1].firm_pop:
                        grid[r][c].pop_firm(firm_i)
                        firm_i.search_market(grid[r][c])[1].add_firm(firm_i)

                grid[r][c].update_w_m_p()

    return grid

print("Original grid (num workers, num firms)")
print(" ")
a = load_grid(5,5,2,1,1.2)

for row in a:
    print(row)
print("#______________________________________________#")
sims = 3
print(f"Grid after {sims} simulations (num workers, num firms)")
print(" ")
b = simulate(sims, a)

for row in b:
    print(row)













