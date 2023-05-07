import random
import numpy as np
from collections import Counter

def read_input_files(enc_file, dict_file, letter_freq_file, letter2_freq_file):
    with open(enc_file, "r") as f:
        enc_text = f.read()
    with open(dict_file, "r") as f:
        dict_text = f.readlines()
    with open(letter_freq_file, "r") as f:
        letter_freq = [float(x.strip().split('\t')[0]) for x in f.readlines()]
    with open(letter2_freq_file, "r") as f:
        letter2_freq = {}
        for line in f.readlines():
            split_line = line.strip().split('\t')
            if len(split_line) < 2:
                continue
            if split_line[1] == 'ZZ':
                break
            letter2_freq[split_line[1]] = float(split_line[0])

    return enc_text, dict_text, letter_freq, letter2_freq

def generate_initial_population(size):
    population = []
    for _ in range(size):
        cipher_alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        real_alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        random.shuffle(real_alphabet)
        solution = [(cipher, real) for cipher, real in zip(cipher_alphabet, real_alphabet)]
        population.append(solution)
    return population

def decipher_text(solution, enc_text):
    deciphered = ""
    for char in enc_text:
        if char.upper() in dict(solution):
            if char.isupper():
                deciphered += dict(solution)[char]
            else:
                deciphered += dict(solution)[char.upper()].lower()
        else:
            deciphered += char
    return deciphered

def fitness(solution, enc_text, dict_text, letter_freq, letter2_freq):
    deciphered = decipher_text(solution, enc_text)
    letter_count = Counter(deciphered)
    total_letters = sum(letter_count.values())
    unigram_score = 0

    for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        unigram_score += abs((letter_count[letter] / total_letters) - letter_freq[i])

    bigram_count = Counter([deciphered[i:i+2] for i in range(len(deciphered) - 1)])
    total_bigrams = sum(bigram_count.values())
    bigram_score = 0
    for bigram, freq in letter2_freq.items():
        bigram_score += abs((bigram_count[bigram] / total_bigrams) - freq)

    words = deciphered.split()
    dict_words = set(word.strip() for word in dict_text)
    dict_word_score = sum(1 for word in words if word.upper() in dict_words)
    return dict_word_score / (unigram_score + bigram_score)

def tournament_selection(population, fitness_values, tournament_size):
    selected = []
    for _ in range(len(population)):
        candidates = random.sample(list(zip(population, fitness_values)), tournament_size)
        winner = max(candidates, key=lambda x: x[1])
        selected.append(winner[0])
    return selected

def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1))
    child1 = parent1[:crossover_point] + [item for item in parent2 if item not in parent1[:crossover_point]]
    child2 = parent2[:crossover_point] + [item for item in parent1 if item not in parent2[:crossover_point]]
    return child1, child2
def mutate(solution, mutation_rate):
    mutated = solution.copy()
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(solution)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated

def genetic_algorithm(max_iterations, population_size, mutation_rate, enc_text, dict_text, letter_freq, letter2_freq, tournament_size=3, no_improvement_threshold=100):
    population = generate_initial_population(population_size)
    best_solution = None
    best_fitness = float("-inf")
    no_improvement_counter = 0
    for iteration in range(max_iterations):
        fitness_values = [fitness(solution, enc_text, dict_text, letter_freq, letter2_freq) for solution in population]
        selected = tournament_selection(population, fitness_values, tournament_size)
        offspring = []

        for i in range(0, len(selected) - (len(selected) % 2), 2):
            child1, child2 = crossover(selected[i], selected[i + 1])
            offspring.append(mutate(child1, mutation_rate))
            offspring.append(mutate(child2, mutation_rate))
        if len(offspring) < len(selected):
            best_selected_solution = max(selected,
                                         key=lambda x: fitness(x, enc_text, dict_text, letter_freq, letter2_freq))
            offspring.append(best_selected_solution)
        population = offspring
        best_solution_iter = max(population, key=lambda x: fitness(x, enc_text, dict_text, letter_freq, letter2_freq))
        best_fitness_iter = fitness(best_solution_iter, enc_text, dict_text, letter_freq, letter2_freq)
        print(f"Iteration: {iteration}, Best solution: {best_solution_iter}, Fitness: {best_fitness_iter}")
        if best_fitness_iter > best_fitness:
            best_solution = best_solution_iter
            best_fitness = best_fitness_iter
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
        if no_improvement_counter >= no_improvement_threshold:
            print("No improvement for a number of iterations, stopping early.")
            break
    return best_solution

def write_output_files(best_solution, enc_text):
    deciphered_text = decipher_text(best_solution, enc_text)

    with open("plain.txt", "w") as f:
        f.write(deciphered_text)

    with open("perm.txt", "w") as f:
        f.write("Cipher\tReal\n")
        for pair in best_solution:
            f.write(f"{pair[0]}\t{pair[1]}\n")

def main():
    enc_file = "enc.txt"
    dict_file = "dict.txt"
    letter_freq_file = "Letter_Freq.txt"
    letter2_freq_file = "Letter2_Freq.txt"
    enc_text, dict_text, letter_freq, letter2_freq = read_input_files(enc_file, dict_file, letter_freq_file, letter2_freq_file)

    max_iterations = 1000
    population_size = 100
    mutation_rate = 0.3
    tournament_size = 3
    no_improvement_threshold = 100

    best_solution = genetic_algorithm(max_iterations, population_size, mutation_rate, enc_text, dict_text, letter_freq, letter2_freq, tournament_size, no_improvement_threshold)

    write_output_files(best_solution, enc_text)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


