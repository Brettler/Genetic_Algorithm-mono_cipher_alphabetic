import random
import string
from collections import Counter

# Load all necessary data
with open('dict.txt', 'r') as f:
    english_words = set(line.strip() for line in f)

with open('Letter_Freq.txt', 'r') as f:
    letter_frequencies = {line.split()[1]: float(line.split()[0]) for line in f}

with open('Letter2_Freq.txt', 'r') as f:
    letter_pair_frequencies = {}
    for line in f:
        if line.split():
            try:
                letter_pair_frequencies[line.split()[1]] = float(line.split()[0])
            except ValueError:
                continue  # skip lines with invalid data

# Genetic representation of solution
def generate_solution():
    return ''.join(random.sample(string.ascii_uppercase, len(string.ascii_uppercase)))

# Fitness function
def fitness(solution, cipher_text):
    deciphered_text = decipher(cipher_text, solution)
    word_count = 0
    total_count = 0

    # split words
    words = deciphered_text.split()
    for word in words:
        if word in english_words:
            word_count += 1
        total_count += 1

    # Calculate frequency fitness
    letter_count = Counter(deciphered_text)
    letter_freq_fitness = sum(abs(letter_frequencies.get(chr(65 + i), 0) - (letter_count.get(chr(65 + i), 0) / len(deciphered_text))) for i in range(26))

    return (word_count / total_count) - letter_freq_fitness

# Decipher function
def decipher(cipher_text, key):
    table = str.maketrans(string.ascii_uppercase, key)
    return cipher_text.translate(table)

# Selection function
def selection(population, scores):
    # Tournament selection
    selected = []
    population_size = len(population)
    for _ in range(population_size):
        i, j = random.sample(range(population_size), 2)
        selected.append(population[i] if scores[i] > scores[j] else population[j])
    return selected

# Crossover function
def crossover(parent1, parent2):
    # Order 1 crossover
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [''] * size
    child[start:end] = parent1[start:end]

    pointer = end
    for gene in parent2[end:] + parent2[:end]:
        if gene not in child:
            while child[pointer] != '':
                pointer = (pointer + 1) % size
            child[pointer] = gene
    return ''.join(child)

# Mutation function
def mutation(solution):
    # Swap mutation
    i, j = random.sample(range(len(solution)), 2)
    mutated = list(solution)
    mutated[i], mutated[j] = mutated[j], mutated[i]
    return ''.join(mutated)

# Generate new solutions
def generate_new_solutions(selected_population, mutation_rate):
    population_size = len(selected_population)
    new_population = []

    for i in range(population_size // 2):
        parent1, parent2 = random.sample(selected_population, 2)
        child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
        if random.random() < mutation_rate:
            child1 = mutation(child1)
        if random.random() < mutation_rate:
            child2 = mutation(child2)
        new_population.extend([child1, child2])
    return new_population

# Genetic Algorithm
def genetic_algorithm(cipher_text, population_size=100, mutation_rate=0.05, max_iterations=1000):
    # Generate initial population
    population = [generate_solution() for _ in range(population_size)]

    for iteration in range(max_iterations):
        scores = [fitness(solution, cipher_text) for solution in population]
        best_solution_iter = population[scores.index(max(scores))]
        best_fitness_iter = max(scores)

        print(f"Iteration: {iteration}, Best solution: {best_solution_iter}, Fitness: {best_fitness_iter}")

        selected = selection(population, scores)
        population = generate_new_solutions(selected, mutation_rate)

    # After all iterations, get the best solution and decipher the text
    final_scores = [fitness(solution, cipher_text) for solution in population]
    best_solution = population[final_scores.index(max(final_scores))]
    deciphered_text = decipher(cipher_text, best_solution)

    # Write the deciphered text into plain.txt
    with open('plain.txt', 'w') as f:
        f.write(deciphered_text)

    # Write the permutation table into perm.txt
    with open('perm.txt', 'w') as f:
        for i in range(26):
            f.write(f"{string.ascii_uppercase[i]} {best_solution[i]}\n")

    return best_solution, max(final_scores)

def main():

    # Usage
    with open('enc.txt', 'r') as f:
        cipher_text = f.read().strip()

    genetic_algorithm(cipher_text, population_size=100, mutation_rate=0.05, max_iterations=1000)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()