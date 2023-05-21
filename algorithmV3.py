# Start of the code
import random
import string
from collections import Counter

# Open the file dict.txt in read mode and load all words
# into a set after converting them to lower case.
with open('dict.txt', 'r') as f:
    english_words = set(line.strip().lower() for line in f)

# Open the file Letter_Freq.txt in read mode and load all letter frequencies
# into a dictionary after converting the letters to lower case.
with open('Letter_Freq.txt', 'r') as f:
    letter_frequencies = {line.split()[1].lower(): float(line.split()[0]) for line in f}

# Open the file Letter2_Freq.txt in read mode and load all letter pair frequencies
# into a dictionary after converting the letters to lower case.
# Ignore lines with invalid data that cannot be converted to a float.
with open('Letter2_Freq.txt', 'r') as f:
    letter_pair_frequencies = {}
    for line in f:
        if line.split():
            try:
                letter_pair_frequencies[line.split()[1].lower()] = float(line.split()[0])
            except ValueError:
                continue  # skip lines with invalid data

def generate_solution():
    """
    :return: random genetic representation of a solution (a random permutation of lowercase English letters)
    """
    return ''.join(random.sample(string.ascii_lowercase, len(string.ascii_lowercase)))

# Fitness function: the higher the score, the closer the solution is to being correct.
def fitness(solution, cipher_text):
    """
    :param solution: permutation of lowercase English letters, such that each letter coding to the real English letter.
    :param cipher_text: Input cipher text we need to decipher.
    :return:  fitness score - weighted combination of the word count fitness, letter frequency fitness,
                                and bigram frequency fitness.
    """
    deciphered_text = decipher(cipher_text, solution)  # Decipher the given cipher text using the provided solution.
    word_count = 0  # number of characters in English words
    total_count = 0  # Total number of characters in all words in the deciphered text
    char_count = Counter(deciphered_text)  # Counts the number of occurrences of each character in the deciphered text.
    list_words_deciphered_text = deciphered_text.split()  # Split the deciphered text into list of words.

    for word in list_words_deciphered_text:  # Iterate on each word in the list.
        if word in english_words:  # If a word is in the English words set,
            word_count += len(word)  # increment the word count by the length of the word.
        total_count += len(word)  # Increment the total count by the length of the word.

    # Calculate the letter frequency fitness by comparing the frequency of each letter in the deciphered text
    # to the known letter frequencies in English.
    #letter_freq_fitness = sum(abs(letter_frequencies.get(chr(97 + i), 0) - (char_count.get(chr(97 + i), 0) / len(deciphered_text))) for i in range(26))
    # Initialize the letter frequency fitness score to 0
    letter_freq_fitness = 0
    all_letters = string.ascii_lowercase
    # Loop over all lowercase English letters
    for current_letter in all_letters:
        # Get the frequency of the current letter in English, default to 0 if not found
        english_letter_frequency = letter_frequencies.get(current_letter, 0)
        # Get the count of the current letter in the deciphered text, default to 0 if not found
        deciphered_letter_count = char_count.get(current_letter, 0)
        # Calculate the frequency of the current letter in the deciphered text
        deciphered_letter_frequency = deciphered_letter_count / len(deciphered_text)
        # Calculate the absolute difference between the letter frequencies
        absolute_difference = abs(english_letter_frequency - deciphered_letter_frequency)
        # Add the absolute difference to the total letter frequency fitness score
        letter_freq_fitness += absolute_difference

    # Calculate the pair_letters_fitness frequency fitness by comparing the frequency of each pair of letters
    # in the deciphered text to the known pair_letters_fitness frequencies in English.
    pair_letters_fitness = 0
    # bigrams = [deciphered_text[i:i+2] for i in range(len(deciphered_text) - 1)]
    bigrams = []  # Initialize an empty list to hold the bigrams.

    # Loop over the indices of the deciphered_text, stopping one before the end.
    for i in range(len(deciphered_text) - 1):
        # Take the current character and the next character.
        bigram = deciphered_text[i:i + 2]

        # Add this pair of characters to the list.
        bigrams.append(bigram)

    for bigram in bigrams:
        if bigram in letter_pair_frequencies:
            pair_letters_fitness += letter_pair_frequencies[bigram]

    fitness_score = 0.4*(word_count / total_count) - 0.2*letter_freq_fitness + 0.2*pair_letters_fitness

    return fitness_score

# Decipher function: replaces each letter in the cipher text with the corresponding letter in the solution.
def decipher(cipher_text, key):
    # table = str.maketrans(string.ascii_lowercase, key)
    # return cipher_text.translate(table)
    table = str.maketrans(string.ascii_lowercase + string.ascii_uppercase,
                          key + key.upper())
    return cipher_text.translate(table)

# Selection function: performs tournament selection to select solutions for the next generation.
def selection(population, scores):
    selected = []  # Create a list to store the best solutions.
    population_size = len(population)  # size of the current population - number of different possible decoding
    for _ in range(population_size):
        # Randomly pick two solutions (i and j) from the population
        i, j = random.sample(range(population_size), 2)
        # The solution with the higher fitness score is considered better.
        selected.append(population[i] if scores[i] > scores[j] else population[j])
    return selected

# Crossover function: performs order 1 crossover to create a child solution from two parent solutions.
def crossover(parent1, parent2):
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


# PMX crossover function: similar to the above, but uses a mapping between parent genes to ensure child genes are not repeated.
def PMX(parent1, parent2):
    print(f"parent1: {parent1}")
    print(f"parent2: {parent2}")
    size = len(parent1)  # Determine the size of the parents
    child = [''] * size  # Initialize the child with empty values

    start, end = sorted(random.sample(range(size), 2))  # Select a random range for crossover

    child[start:end] = parent1[start:end]  # Copy the selected range from parent1 to the child

    # Create a mapping between the genes in the selected range of parent1 and parent2
    #mapping = dict(zip(parent1[start:end], parent2[start:end]))

    # We first slice parent1 and parent2 to obtain the selected range from each
    slice_parent1 = parent1[start:end]
    slice_parent2 = parent2[start:end]

    # Then we use the zip function, which combines two iterables element-wise.
    # This will result in a list of tuples where the i-th tuple contains
    # the i-th element from each of the argument sequences or iterables.
    zipped_parents = zip(slice_parent1, slice_parent2)

    # We then convert the zipped result into a dictionary. In this dictionary,
    # each key-value pair corresponds to a character in the selected range of parent1 mapped to the
    # corresponding character in the selected range of parent2.
    mapping = dict(zipped_parents)
    #print(f"mapping: {mapping}")
    # Fill the rest of the child solution.
    for i, letter in enumerate(parent2):  # Iterate over the letters of parent2
        # Check if the letter is outside the selected range because in the selected range the child already have
        # the letters from parent1.
        #print(f"letter in parent2: {letter}")
        #savelastletter = letter
        if i not in range(start, end):
            # If the letter we are current iterate is already in the mapping it means this letter is in the parent1.
            # If this letter is in parent1 we wont add it to the child because the child already contain it.
            while letter in mapping:  # Check if the letter is already in the mapping
                #savelastletter = letter
                letter = mapping[letter]  #  Replace the letter with the corresponding letter from parent2 based on the mapping

                #print(f"{letter} =mapping[{savelastletter}]")
            child[i] = letter  # Assign the letter to the child
    #print(f"child is {''.join(child)}")
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


# Mutation function: performs a swap mutation on a solution if a random number is less than the mutation rate.
def mutation(solution, mutation_rate):
    # Swap mutation only if mutation rate condition is satisfied
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(solution)), 2)
        mutated = list(solution)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return ''.join(mutated)
    return solution

# Generate new solutions for the next generation by applying crossover and mutation to the selected population.
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
def genetic_algorithm(cipher_text, population_size=100, max_mutation_rate=0.5, min_mutation_rate=0.05, max_iterations=200, elitism=True, fitness_stagnation_threshold=20):
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

def calculate_accuracy(true_coding_file, results_file):
    true_coding = {}
    with open(true_coding_file, 'r') as f:
        for line in f:
            line = line.strip().split()
            true_coding[line[0]] = line[1]

    total_letters = 0
    correct_substitutions = 0

    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip().split()
            letter = line[0]
            result = line[1]

            total_letters += 1
            if letter in true_coding and true_coding[letter] == result:
                correct_substitutions += 1

    accuracy = (correct_substitutions / total_letters) * 100
    return accuracy

def main():
    # Usage
    with open('enc2.txt', 'r') as f:
        cipher_text = f.read().strip()

    genetic_algorithm(cipher_text.lower(), population_size=200, max_mutation_rate=0.4, min_mutation_rate=0.05, max_iterations=500, elitism=True)
    true_coding_file = 'true_perm.txt'  # Replace with the actual file name and path
    results_file = 'perm.txt'  # Replace with the actual file name and path

    accuracy = calculate_accuracy(true_coding_file, results_file)
    print(f"Accuracy: {accuracy:.2f}%")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    exit(1)
