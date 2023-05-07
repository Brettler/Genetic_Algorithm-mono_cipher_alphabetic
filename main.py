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

def generate_initial_population(population_size):
    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    initial_population = []

    for _ in range(population_size):
        # Shuffle the alphabet list to create a random mapping.
        shuffled_alphabet = alphabet.copy()
        random.shuffle(shuffled_alphabet)

        # Create a mapping of each letter in the alphabet to a unique letter from the shuffled alphabet.
        solution = list(zip(alphabet, shuffled_alphabet))
        initial_population.append(solution)

    return initial_population

def decipher_text(solution, enc_text):
    # Initialize an empty string to store the deciphered text
    deciphered = ""
    # Iterate over each character in the encrypted text
    for char in enc_text:
        # Check if the uppercase version of the character is in the solution dictionary.
        if char.upper() in dict(solution):
            # If the character is uppercase, add the deciphered uppercase character to the deciphered string
            if char.isupper():
                deciphered += dict(solution)[char]
            # If the character is lowercase, add the deciphered lowercase character to the deciphered string
            else:
            # If the character is not in the solution dictionary (e.g., punctuation or whitespace), add it as-is to the deciphered string
                deciphered += dict(solution)[char.upper()].lower()
        else:
            deciphered += char
    # Return the complete deciphered text
    return deciphered

def fitness(solution, enc_text, dict_text, letter_freq, letter2_freq):
    # Decrypt the encrypted text using the given solution.
    deciphered = decipher_text(solution, enc_text)

    # Unigram fitness score.
    # Count the occurrences of each letter in the deciphered text.
    letter_count = Counter(deciphered)
    # Calculate the total number of letters in the deciphered text.
    total_letters = sum(letter_count.values())
    unigram_score = 0

    # Compare the frequency of each letter in the deciphered text to the known letter frequency (letter_freq).
    for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        unigram_score += abs((letter_count[letter] / total_letters) - letter_freq[i])

    # Bigram fitness score
    # Count the occurrences of each bigram (pair of letters) in the deciphered text
    bigram_count = Counter([deciphered[i:i+2] for i in range(len(deciphered) - 1)])
    # Calculate the total number of bigrams in the deciphered text
    total_bigrams = sum(bigram_count.values())
    bigram_score = 0

    # Compare the frequency of each bigram in the deciphered text to the known bigram frequency (letter2_freq)
    for bigram, freq in letter2_freq.items():
        bigram_score += abs((bigram_count[bigram] / total_bigrams) - freq)

    # Dictionary words fitness score
    # Split the deciphered text into individual words
    words = deciphered.split()
    # Create a set of dictionary words from dict_text
    dict_words = set(word.strip() for word in dict_text)
    # Count the number of deciphered words that are present in the dictionary
    dict_word_score = sum(1 for word in words if word.upper() in dict_words)

    # Calculate the overall fitness score by dividing the dictionary words score by the sum of the unigram and bigram scores
    return dict_word_score / (unigram_score + bigram_score)

def tournament_selection(population, fitness_values, tournament_size):
    # Initialize an empty list to store the selected individuals
    selected = []
    # Perform selection process for each individual in the population
    for _ in range(len(population)):
        # Randomly pick a specified number of individuals (tournament_size) from the population
        # and their corresponding fitness values
        candidates = random.sample(list(zip(population, fitness_values)), tournament_size)
        # Find the candidate with the highest fitness value
        winner = max(candidates, key=lambda x: x[1])
        # Add the winning individual (with the highest fitness value) to the selected list
        selected.append(winner[0])
    # Return the list of selected individuals
    return selected

def crossover(parent1, parent2):
    # Generate a random crossover point
    crossover_point = random.randint(0, len(parent1))

    # Create the first child by combining the first part of parent1 up to the crossover point
    first_part_child1 = parent1[:crossover_point]

    # Create the remaining unique elements from parent2 in their original order
    remaining_part_child1 = [item for item in parent2 if item not in first_part_child1]

    # Combine the two parts to form the first child
    child1 = first_part_child1 + remaining_part_child1

    # Create the first part of the second child by taking elements from parent2 up to the crossover point
    first_part_child2 = parent2[:crossover_point]

    # Create the remaining unique elements from parent1 in their original order
    remaining_part_child2 = [item for item in parent1 if item not in first_part_child2]

    # Combine the two parts to form the second child
    child2 = first_part_child2 + remaining_part_child2

    # Return the two created children
    return child1, child2
def mutate(solution, mutation_rate):
    # Make a copy of the solution to avoid modifying the original
    mutated = solution.copy()

    # Generate a random float between 0 and 1
    mutation_probability = random.random()

    # Check if the random float is less than the mutation rate, in which case mutation will occur
    if mutation_probability < mutation_rate:
        # Randomly choose two indices in the solution
        i, j = random.sample(range(len(solution)), 2)

        # Swap the elements at the chosen indices
        temp = mutated[i]
        mutated[i] = mutated[j]
        mutated[j] = temp

    # Return the mutated solution (or the original solution if no mutation occurred)
    return mutated

def genetic_algorithm(max_iterations, population_size, mutation_rate, enc_text, dict_text, letter_freq, letter2_freq, tournament_size=3, no_improvement_threshold=100):
    # Generate the initial population of solutions
    population = generate_initial_population(population_size)

    # Initialize variables for tracking the best solution found so far
    best_solution = None
    best_fitness = float("-inf")
    no_improvement_counter = 0

    # Loop for the maximum number of iterations
    for iteration in range(max_iterations):
        # Calculate the fitness of each solution in the population
        fitness_values = [fitness(solution, enc_text, dict_text, letter_freq, letter2_freq) for solution in population]

        # Select parents for mating using tournament selection
        selected = tournament_selection(population, fitness_values, tournament_size)

        # Create offspring through crossover and mutation
        offspring = []

        # Loop over pairs of selected parents to create offspring.
        for i in range(0, len(selected) - (len(selected) % 2), 2):
            # Generate two children by performing crossover on the selected parents.
            child1, child2 = crossover(selected[i], selected[i + 1])

            # Mutate the children with a certain probability using the mutate function.
            mutated_child1 = mutate(child1, mutation_rate)
            mutated_child2 = mutate(child2, mutation_rate)

            # Add mutated children to the offspring list.
            offspring.append(mutated_child1)
            offspring.append(mutated_child2)

        # If the number of offspring is less than the number of selected parents, add the best parent to the offspring.
        if len(offspring) < len(selected):
            best_selected_solution = max(selected, key=lambda x: fitness(x, enc_text, dict_text, letter_freq, letter2_freq))
            offspring.append(best_selected_solution)

        # Update the population with the offspring.
        population = offspring

        # Find the best solution for this iteration
        best_solution_iter = max(population, key=lambda x: fitness(x, enc_text, dict_text, letter_freq, letter2_freq))
        best_fitness_iter = fitness(best_solution_iter, enc_text, dict_text, letter_freq, letter2_freq)

        # Print the iteration number, best solution, and best fitness.
        print(f"Iteration: {iteration}, Best solution: {best_solution_iter}, Fitness: {best_fitness_iter}")

        # Update the best solution found so far if the current iteration's best solution is better.
        if best_fitness_iter > best_fitness:
            best_solution = best_solution_iter
            best_fitness = best_fitness_iter
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        # If the best solution has not improved for a certain number of iterations, terminate the loop.
        if no_improvement_counter >= no_improvement_threshold:
            print("No improvement for a number of iterations, stopping early.")
            break

    # Return the best solution found.
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


