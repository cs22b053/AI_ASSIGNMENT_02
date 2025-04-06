import time
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# --- TSP Setup Functions ---
def generate_cities(n, width=100, height=100):
    """Generate n cities with random (x, y) coordinates."""
    return [(random.uniform(0, width), random.uniform(0, height)) for _ in range(n)]

def total_distance(tour, cities):
    """Compute the total distance of the tour (using Euclidean distance)."""
    dist = 0
    for i in range(len(tour)):
        j = (i + 1) % len(tour)
        x1, y1 = cities[tour[i]]
        x2, y2 = cities[tour[j]]
        dist += math.hypot(x2 - x1, y2 - y1)
    return dist

def swap_two_cities(tour):
    """Swap two cities in the tour to create a neighbor solution."""
    new_tour = tour.copy()
    i, j = random.sample(range(len(tour)), 2)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

# --- Hill Climbing Implementation with History Capturing ---
def hill_climbing(cities, timeout=600, record_interval=100):
    current = list(range(len(cities)))
    random.shuffle(current)
    current_cost = total_distance(current, cities)
    start_time = time.time()
    
    # History will store tuples: (tour, cost)
    history = [(current.copy(), current_cost)]
    no_improve_counter = 0
    max_no_improve = 10000
    iteration = 0
    
    while time.time() - start_time < timeout and no_improve_counter < max_no_improve:
        iteration += 1
        neighbor = swap_two_cities(current)
        neighbor_cost = total_distance(neighbor, cities)
        if neighbor_cost < current_cost:
            current, current_cost = neighbor, neighbor_cost
            no_improve_counter = 0
            # Record state on improvement
            history.append((current.copy(), current_cost))
        else:
            no_improve_counter += 1
        # Also record state periodically
        if iteration % record_interval == 0:
            history.append((current.copy(), current_cost))
    
    elapsed = time.time() - start_time
    return current, current_cost, elapsed, history

# --- Simulated Annealing Implementation with History Capturing ---
def simulated_annealing(cities, timeout=600, initial_temp=1000, cooling_rate=0.995, record_interval=100):
    current = list(range(len(cities)))
    random.shuffle(current)
    current_cost = total_distance(current, cities)
    best = current.copy()
    best_cost = current_cost
    temp = initial_temp
    start_time = time.time()
    
    history = [(current.copy(), current_cost)]
    iteration = 0
    
    while time.time() - start_time < timeout and temp > 1e-8:
        iteration += 1
        neighbor = swap_two_cities(current)
        neighbor_cost = total_distance(neighbor, cities)
        delta = neighbor_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current, current_cost = neighbor, neighbor_cost
            if current_cost < best_cost:
                best, best_cost = current.copy(), current_cost
                history.append((best.copy(), best_cost))
        temp *= cooling_rate
        
        if iteration % record_interval == 0:
            history.append((best.copy(), best_cost))
    
    elapsed = time.time() - start_time
    return best, best_cost, elapsed, history

# --- Plotting & GIF Creation ---
def plot_tour(cities, tour, cost, ax):
    """Plot the TSP tour on the given axis."""
    ax.clear()
    # Prepare x and y coordinates in tour order (and close the loop)
    x_coords = [cities[i][0] for i in tour] + [cities[tour[0]][0]]
    y_coords = [cities[i][1] for i in tour] + [cities[tour[0]][1]]
    ax.plot(x_coords, y_coords, marker='o', color='blue')
    ax.set_title(f"Cost: {cost:.2f}")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

def create_animation(cities, history, filename='animation.gif'):
    """Create and save an animated GIF from the recorded history of tours."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def update(frame):
        tour, cost = history[frame]
        plot_tour(cities, tour, cost, ax)
        ax.set_title(f"Iteration: {frame}, Cost: {cost:.2f}")
        return ax,
    
    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=200, blit=False)
    ani.save(filename, writer='pillow', fps=5)
    plt.close(fig)

# --- Running Experiments and Saving GIFs ---
def run_experiment(n_runs=5, n_cities=20, timeout=600):
    # Generate a single TSP instance (for both algorithms to compare on the same cities)
    cities = generate_cities(n_cities)
    
    # Run Hill Climbing experiments
    print("Running Hill Climbing:")
    hc_times = []
    for run in range(n_runs):
        best, cost, time_taken, history = hill_climbing(cities, timeout)
        hc_times.append(time_taken)
        print(f"HC Run {run+1}: Cost = {cost:.2f}, Time = {time_taken:.2f} sec, Steps recorded: {len(history)}")
        gif_filename = f"hc_run_{run+1}.gif"
        create_animation(cities, history, filename=gif_filename)
        print(f"Saved GIF: {gif_filename}")
    
    avg_hc_time = sum(hc_times) / len(hc_times)
    print(f"Average Hill Climbing Time: {avg_hc_time:.2f} sec")
    
    # Run Simulated Annealing experiments
    print("\nRunning Simulated Annealing:")
    sa_times = []
    for run in range(n_runs):
        best, cost, time_taken, history = simulated_annealing(cities, timeout)
        sa_times.append(time_taken)
        print(f"SA Run {run+1}: Cost = {cost:.2f}, Time = {time_taken:.2f} sec, Steps recorded: {len(history)}")
        gif_filename = f"sa_run_{run+1}.gif"
        create_animation(cities, history, filename=gif_filename)
        print(f"Saved GIF: {gif_filename}")
    
    avg_sa_time = sum(sa_times) / len(sa_times)
    print(f"Average Simulated Annealing Time: {avg_sa_time:.2f} sec")

# --- Main Execution ---
if __name__ == '__main__':
    run_experiment(n_runs=10, n_cities=50, timeout=600)
