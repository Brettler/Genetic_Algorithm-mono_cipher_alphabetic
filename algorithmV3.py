# Start of the code
import random
import string
from collections import Counter

# Load all necessary data
with open('dict.txt', 'r') as f:
    english_words = set(line.strip().lower() for line in f)  # Change to lower case

with open('Letter_Freq.txt', 'r') as f:
    letter_frequencies = {line.split()[1].lower(): float(line.split()[0]) for line in f}  # Change to lower case

with open('Letter2_Freq.txt', 'r') as f:
    letter_pair_frequencies = {}
    for line in f:
        if line.split():
            try:
                letter_pair_frequencies[line.split()[1].lower()] = float(line.split()[0])  # Change to lower case
            except ValueError:
                continue  # skip lines with invalid data

# Genetic representation of solution
def generate_solution():
    return ''.join(random.sample(string.ascii_lowercase, len(string.ascii_lowercase)))  # Change to lower case

# Fitness function
def fitness(solution, cipher_text):
    deciphered_text = decipher(cipher_text, solution)
    word_count = 0
    total_count = 0
    char_count = Counter(deciphered_text)

    # split words
    words = deciphered_text.split()
    for word in words:
        if word in english_words:
            word_count += len(word)  # weight by length of word
        total_count += len(word)

    # Calculate frequency fitness
    letter_freq_fitness = sum(abs(letter_frequencies.get(chr(97 + i), 0) - (char_count.get(chr(97 + i), 0) / len(deciphered_text))) for i in range(26))

    # Adding bigram frequency
    bigram_fitness = 0
    bigrams = [deciphered_text[i:i+2] for i in range(len(deciphered_text) - 1)]
    for bigram in bigrams:
        if bigram in letter_pair_frequencies:
            bigram_fitness += letter_pair_frequencies[bigram]

    # Normalize by length of text and weight each factor
    fitness_score = 0.4*(word_count / total_count) - 0.4*letter_freq_fitness + 0.2*bigram_fitness
    score_maybe = word_count / total_count

    return fitness_score

# Decipher function
def decipher(cipher_text, key):
    table = str.maketrans(string.ascii_lowercase, key)  # Change to lower case
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


# Big chance it is the best function:
def PMX(parent1, parent2):
    # Determine the size of the parents
    size = len(parent1)

    # Select a random range for crossover
    start, end = sorted(random.sample(range(size), 2))

    # Initialize the child with empty values
    child = [''] * size

    # Copy the selected range from parent1 to the child
    child[start:end] = parent1[start:end]

    # Create a mapping between the genes in the selected range of parent1 and parent2
    mapping = dict(zip(parent1[start:end], parent2[start:end]))

    # Iterate over the genes of parent2
    for i, gene in enumerate(parent2):
        # Check if the gene is outside the selected range
        if i not in range(start, end):
            # Check if the gene is already in the mapping
            while gene in mapping:
                # Replace the gene with the corresponding gene from parent2 based on the mapping
                gene = mapping[gene]
            # Assign the gene to the child
            child[i] = gene

    # Return the child as a string
    return ''.join(child)

def CX(parent1, parent2):
    size = len(parent1)
    child = [''] * size

    index = 0
    while '' in child:
        cycle_start = parent1[index]
        while True:
            child[index] = parent1[index]
            index = parent1.index(parent2[index])
            if parent1[index] == cycle_start:
                break
        index = (child.index('') if '' in child else None)

    return ''.join(child)


# Mutation function with adaptive mutation rate
def mutation(solution, mutation_rate):
    # Swap mutation only if mutation rate condition is satisfied
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(solution)), 2)
        mutated = list(solution)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return ''.join(mutated)
    return solution

# Generate new solutions
def generate_new_solutions(selected_population, mutation_rate):
    population_size = len(selected_population)
    new_population = []

    for i in range(population_size // 2):
        parent1, parent2 = random.sample(selected_population, 2)
        # Choose crossover method
        # child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
        child1, child2 = PMX(parent1, parent2), PMX(parent2, parent1)
        # child1, child2 = CX(parent1, parent2), CX(parent2, parent1)



        child1 = mutation(child1, mutation_rate)
        child2 = mutation(child2, mutation_rate)
        new_population.extend([child1, child2])
    return new_population

# Genetic Algorithm
def genetic_algorithm(cipher_text, population_size=100, max_mutation_rate=0.5, min_mutation_rate=0.05, max_iterations=1000, elitism=True, fitness_stagnation_threshold=500):
    # Generate initial population
    population = [generate_solution() for _ in range(population_size)]
    best_score = float('-inf')
    best_solution = ''
    stagnation_counter = 0  # Counter to track the number of iterations with no improvement

    for iteration in range(max_iterations):
        mutation_rate = max_mutation_rate - (max_mutation_rate - min_mutation_rate) * (iteration/max_iterations)
        scores = [fitness(solution, cipher_text) for solution in population]
        max_score = max(scores)
        max_index = scores.index(max_score)
        if max_score > best_score:
            best_score = max_score
            best_solution = population[max_index]
            stagnation_counter = 0  # Reset the counter if there is an improvement
        else:
            stagnation_counter += 1  # Increment the counter if there is no improvement

        print(f"Iteration: {iteration}, Best solution: {best_solution}, Fitness: {best_score}, Mutation rate: {mutation_rate}")

        if stagnation_counter >= fitness_stagnation_threshold:
            print(f"No improvement in fitness score for {fitness_stagnation_threshold} iterations. Stopping the algorithm.")
            break



        selected = selection(population, scores)
        population = generate_new_solutions(selected, mutation_rate)

        # Elitism - Ensure best solution is always in population
        if elitism and best_solution not in population:
            population[random.randint(0, population_size-1)] = best_solution

    # After all iterations, decipher the text with best solution
    deciphered_text = decipher(cipher_text, best_solution)

    # Write the deciphered text into plain.txt
    with open('plain.txt', 'w') as f:
        f.write(deciphered_text)

    # Write the permutation table into perm.txt
    with open('perm.txt', 'w') as f:
        for i in range(26):
            f.write(f"{string.ascii_lowercase[i]} {best_solution[i]}\n")  # Change to lower case

    return best_solution, best_score

def main():
    # Usage
    with open('enc2.txt', 'r') as f:
        cipher_text = f.read().strip().lower()  # Change to lower case

    genetic_algorithm(cipher_text, population_size=120, max_mutation_rate=0.5, min_mutation_rate=0.05, max_iterations=500, elitism=True)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
