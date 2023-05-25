import random
import string
from collections import Counter
from collections import deque

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


def init_generate_solution():
    """
    :return: random genetic representation of a solution (a random permutation of lowercase English letters)
    """
    return ''.join(random.sample(string.ascii_lowercase, len(string.ascii_lowercase)))


# Fitness function: the higher the score, the closer the solution is to being correct.
def fitness(solutions, cipher_text):
    """
    :param solutions: a list of solutions. Each solution is a permutation of lowercase English letters, such that each letter coding to the real English letter.
    :param cipher_text: Input cipher text we need to decipher.
    :return:  fitness score - weighted combination of the word count fitness, letter frequency fitness,
                                and bigram frequency fitness.
    """
    fitness_scores = []
    letter_freq_dif_list = []
    for solution in solutions:
        deciphered_text = decipher(cipher_text, solution)  # Decipher the given cipher text using the provided solution.
        word_count = 0  # number of characters in English words
        total_count = 0  # Total number of characters in all words in the deciphered text
        char_count = Counter(
            deciphered_text)  # Counts the number of occurrences of each character in the deciphered text.
        list_words_deciphered_text = deciphered_text.split()  # Split the deciphered text into list of words.

        new_text = ""
        for e in deciphered_text:
            if e.isalpha():
                new_text += e
        deciphered_text = new_text

        for word in list_words_deciphered_text:  # Iterate on each word in the list.
            clean_word = "".join(e for e in word if e.isalpha())
            if clean_word in english_words:  # If a word is in the English words set,
                word_count += len(clean_word)  # increment the word count by the length of the word.
            total_count += len(clean_word)  # Increment the total count by the length of the word.

        # Calculate the letter frequency fitness by comparing the frequency of each letter in the deciphered text
        # to the known letter frequencies in English.
        # Initialize the letter frequency fitness score to 0
        letter_freq_fitness = 0
        all_letters = string.ascii_lowercase  # 'abcdefg....'
        letter_freq_dif = {}
        letter_correct = 0
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
            letter_freq_dif[current_letter] = absolute_difference
            # Add the absolute difference to the total letter frequency fitness score
            letter_freq_fitness += absolute_difference
            if absolute_difference == 0:
                letter_correct += english_letter_frequency
        letter_freq_dif_list.append(letter_freq_dif)
        # Calculate the pair_letters_fitness frequency fitness by comparing the frequency of each pair of letters
        # in the deciphered text to the known pair_letters_fitness frequencies in English.
        pair_letters_fitness = 0
        # bigrams = [deciphered_text[i:i+2] for i in range(len(deciphered_text) - 1)]
        bigrams = []  # Initialize an empty list to hold the bigrams.

        # # Loop over the indices of the deciphered_text, stopping one before the end.
        # for i in range(len(deciphered_text) - 1):
        #     # Take the current character and the next character.
        #     bigram = deciphered_text[i:i + 2]
        #
        #     # Add this pair of characters to the list.
        #     bigrams.append(bigram)
        #
        # for bigram in bigrams:
        #     if bigram in letter_pair_frequencies:
        #         pair_letters_fitness += letter_pair_frequencies[bigram]
        pair_letters_correct = 0
        bigram_count = Counter([deciphered_text[i:i + 2] for i in range(len(deciphered_text) - 1)])
        for bigram, count in bigram_count.items():
            # Get the frequency of the current bigram in English, default to 0 if not found
            english_bigram_frequency = letter_pair_frequencies.get(bigram, 0)
            # Calculate the frequency of the current bigram in the deciphered text
            deciphered_bigram_frequency = count / (len(deciphered_text) - 1)
            # Calculate the absolute difference between the bigram frequencies
            absolute_difference = abs(english_bigram_frequency - deciphered_bigram_frequency)
            # Add the absolute difference to the total pair_letters_fitness score
            pair_letters_fitness += absolute_difference
            if absolute_difference == 0:
                pair_letters_correct += english_bigram_frequency

        fitness_score = (word_count / total_count) - 0.5 * letter_freq_fitness - 0.5 * pair_letters_fitness \
                        + 0.5*pair_letters_correct/(len(deciphered_text) - 1) + 0.5*letter_correct/len(deciphered_text)
        # append the fitness_score of this solution to the fitness_scores list
        fitness_scores.append(fitness_score)

    return fitness_scores, letter_freq_dif_list


def simplified_fitness(solutions, cipher_text):
    """
    :param solution: permutation of lowercase English letters, such that each letter coding to the real English letter.
    :param cipher_text: Input cipher text we need to decipher.
    :return:  fitness score - weighted combination of the word count fitness, letter frequency fitness,
                                and bigram frequency fitness.
    """

    fitness_scores = []
    letter_freq_dif_list = []
    for solution in solutions:
        deciphered_text = decipher(cipher_text, solution)  # Decipher the given cipher text using the provided solution.
        word_count = 0  # number of characters in English words
        total_count = 0  # Total number of characters in all words in the deciphered text
        char_count = Counter(
            deciphered_text)  # Counts the number of occurrences of each character in the deciphered text.
        list_words_deciphered_text = deciphered_text.split()  # Split the deciphered text into list of words.

        new_text = ""
        for e in deciphered_text:
            if e.isalpha():
                new_text += e
        deciphered_text = new_text

        for word in list_words_deciphered_text:  # Iterate on each word in the list.
            clean_word = "".join(e for e in word if e.isalpha())
            if clean_word in english_words:  # If a word is in the English words set,
                word_count += len(clean_word)  # increment the word count by the length of the word.
            total_count += len(clean_word)  # Increment the total count by the length of the word.

        # Calculate the letter frequency fitness by comparing the frequency of each letter in the deciphered text
        # to the known letter frequencies in English.
        # Initialize the letter frequency fitness score to 0
        letter_freq_fitness = 0
        all_letters = string.ascii_lowercase  # 'abcdefg....'
        letter_freq_dif = {}
        # Loop over all lowercase English letters
        for current_letter in all_letters:
            # Get the frequency of the current letter in English, default to 0 if not found
            english_letter_frequency = letter_frequencies.get(current_letter, 0)
            # Get the count of the current letter in the deciphered text, default to 0 if not found
            deciphered_letter_count = char_count.get(current_letter, 0)
            # Calculate the frequency of the current letter in the deciphered text
            ################################ Need to swich len(deciphered_text) by the length without spaces (just check the letters) #####################3
            deciphered_letter_frequency = deciphered_letter_count / len(deciphered_text)
            # Calculate the absolute difference between the letter frequencies
            absolute_difference = abs(english_letter_frequency - deciphered_letter_frequency)
            letter_freq_dif[current_letter] = absolute_difference
            # Add the absolute difference to the total letter frequency fitness score
            letter_freq_fitness += absolute_difference
        letter_freq_dif_list.append(letter_freq_dif)
        # Calculate the pair_letters_fitness frequency fitness by comparing the frequency of each pair of letters
        # in the deciphered text to the known pair_letters_fitness frequencies in English.
        pair_letters_fitness = 0
        # bigrams = [deciphered_text[i:i+2] for i in range(len(deciphered_text) - 1)]
        bigrams = []  # Initialize an empty list to hold the bigrams.

        # Loop over the indices of the deciphered_text, stopping one before the end.
        for i in range(len(deciphered_text) - 1):
            ###################################################### We need to check if spaces are included ####################################
            # Take the current character and the next character.
            bigram = deciphered_text[i:i + 2]

            # Add this pair of characters to the list.
            bigrams.append(bigram)

        for bigram in bigrams:
            if bigram in letter_pair_frequencies:
                pair_letters_fitness += letter_pair_frequencies[bigram]

        fitness_score = (word_count / total_count) - 0.5 * letter_freq_fitness - 0.5 * pair_letters_fitness
        # append the fitness_score of this solution to the fitness_scores list
        fitness_scores.append(fitness_score)

    return fitness_scores, letter_freq_dif_list


# Decipher function: replaces each letter in the cipher text with the corresponding letter in the solution.
def decipher(cipher_text, perm_rules_decoder):
    # table = str.maketrans(string.ascii_lowercase, key)
    # return cipher_text.translate(table)

    table = str.maketrans(string.ascii_lowercase + string.ascii_uppercase,
                          perm_rules_decoder + perm_rules_decoder.upper())
    return cipher_text.translate(table)


def decipher_convert_format(cipher_text, perm_rules_decoder):
    # Stores a list of boolean values indicating whether each character in the cipher text is uppercase.
    original_case = [c.isupper() for c in cipher_text]
    cipher_text = cipher_text.lower()
    table = str.maketrans(string.ascii_lowercase, perm_rules_decoder)
    deciphered_text = cipher_text.translate(table)
    # deciphered text and converts each character to uppercase if the corresponding character
    # in the original cipher text was uppercase
    deciphered_text = ''.join(
        c.upper() if original_case[i] else c
        for i, c in enumerate(deciphered_text)
    )
    return deciphered_text


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
# def crossover(parent1, parent2):
#     size = len(parent1)
#     start, end = sorted(random.sample(range(size), 2))
#     child = [''] * size
#     child[start:end] = parent1[start:end]
#
#     pointer = end
#     for gene in parent2[end:] + parent2[:end]:
#         if gene not in child:
#             while child[pointer] != '':
#                 pointer = (pointer + 1) % size
#             child[pointer] = gene
#     return ''.join(child)


# PMX crossover function: similar to the above, but uses a mapping between parent genes to ensure child genes are not repeated.
def PMX(parent1, parent2):
    # print(f"parent1: {parent1}")
    # print(f"parent2: {parent2}")
    size = len(parent1)  # Determine the size of the parents
    child = [''] * size  # Initialize the child with empty values

    start, end = sorted(random.sample(range(size), 2))  # Select a random range for crossover

    child[start:end] = parent1[start:end]  # Copy the selected range from parent1 to the child

    # Create a mapping between the genes in the selected range of parent1 and parent2
    # mapping = dict(zip(parent1[start:end], parent2[start:end]))

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
    # print(f"mapping: {mapping}")
    # Fill the rest of the child solution.
    for i, letter in enumerate(parent2):  # Iterate over the letters of parent2
        # Check if the letter is outside the selected range because in the selected range the child already have
        # the letters from parent1.
        # print(f"letter in parent2: {letter}")
        # savelastletter = letter
        if i not in range(start, end):
            # If the letter we are current iterate is already in the mapping it means this letter is in the parent1.
            # If this letter is in parent1 we wont add it to the child because the child already contain it.
            while letter in mapping:  # Check if the letter is already in the mapping
                # savelastletter = letter
                letter = mapping[
                    letter]  # Replace the letter with the corresponding letter from parent2 based on the mapping

                # print(f"{letter} =mapping[{savelastletter}]")
            child[i] = letter  # Assign the letter to the child
    # print(f"child is {''.join(child)}")
    # Return the child as a string
    return ''.join(child)


# def CX(parent1, parent2):
#     size = len(parent1)
#     child = [''] * size
#
#     index = 0
#     while '' in child:
#         cycle_start = parent1[index]
#         while True:
#             child[index] = parent1[index]
#             index = parent1.index(parent2[index])
#             if parent1[index] == cycle_start:
#                 break
#         index = (child.index('') if '' in child else None)
#
#     return ''.join(child)


"""
# Mutation function: performs a swap mutation on a solution if a random number is less than the mutation rate.
def mutation(solution, mutation_rate, letter_freq_opt):
    # Swap mutation only if mutation rate condition is satisfied
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(solution)), 2)
        mutated = list(solution)
        print(f"mutated: {mutated}")
        mutated[i], mutated[j] = mutated[j], mutated[i]
        print(f"mutated after: {''.join(mutated)}")
        return ''.join(mutated)
    return solution
"""


def mutation(solution, mutation_rate, max_letter, min_letter):
    # Swap mutation only if mutation rate condition is satisfied
    if random.random() < mutation_rate:
        i = solution.index(max_letter)  # find the index of max_letter
        idx_min = solution.index(min_letter)  # find the index of min_letter
        # choose a random index different from i and not equal to the index of min_letter
        j = random.choice([k for k in range(len(solution)) if k != i and k != idx_min])
        mutated = list(solution)
        mutated[i], mutated[j] = mutated[j], mutated[i]  # swap letters at positions i and j
        return ''.join(mutated)
    return solution


# Generate new solutions for the next generation by applying crossover and mutation to the selected population.
def generate_new_solutions(selected_population, mutation_rate, letter_freq_opt):
    population_size = len(selected_population)
    new_population = []

    for i in range(population_size // 2):
        parent1, parent2 = random.sample(selected_population, 2)
        # Choose crossover method
        # child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
        child1, child2 = PMX(parent1, parent2), PMX(parent2, parent1)
        # child1, child2 = CX(parent1, parent2), CX(parent2, parent1)
        parent1_idx = selected_population.index(parent1)
        parent2_idx = selected_population.index(parent2)
        max_letter_daddy = max(letter_freq_opt[parent1_idx].items(), key=lambda x: x[1])[0]
        min_letter_daddy = min(letter_freq_opt[parent1_idx].items(), key=lambda x: x[1])[0]

        max_letter_mommy = max(letter_freq_opt[parent2_idx].items(), key=lambda x: x[1])[0]
        min_letter_mommy = min(letter_freq_opt[parent1_idx].items(), key=lambda x: x[1])[0]

        # print(f"max_letter_dady: {max_letter_dady}")
        # print(f"max_letter_momy: {max_letter_momy}")
        child1 = mutation(child1, mutation_rate, max_letter_daddy, min_letter_daddy)
        child2 = mutation(child2, mutation_rate, max_letter_mommy, min_letter_mommy)
        new_population.extend([child1, child2])
    return new_population


def local_optimization(population, cipher_text, N, lamarkian=False):
    # Initialize the best solutions and scores with the current population
    best_solutions = list(population)
    # Calculate the fitness scores for the entire population
    best_scores, letter_freq_opt = fitness(population, cipher_text)

    # Iterate over each solution in the population
    for idx in range(len(population)):
        solution = population[idx]

        # Perform local optimization for the specified number of iterations (N)
        for _ in range(N):
            i, j = random.sample(range(len(solution)), 2)  # Select two random positions
            candidate = list(solution)
            candidate[i], candidate[j] = candidate[j], candidate[i]  # Swap the selected positions
            candidate = ''.join(candidate)
            # print(f"canidate is: {candidate}")
            population[idx] = candidate  # Update the solution in the population
            # print(f"canidate is: {candidate}")
        # Calculate the fitness scores for the entire population

        candidate_scores, new_letter_freq_opt = fitness([population[idx]],
                                                        cipher_text)  # Get the fitness score of the candidate solution

        # Update the best score and solution if the candidate score is higher
        if candidate_scores[0] > best_scores[idx]:
            best_scores[idx] = candidate_scores[0]
            if lamarkian:
                best_solutions[idx] = population[idx]
                letter_freq_opt[idx] = new_letter_freq_opt[0]

    return best_solutions, best_scores, letter_freq_opt


# Genetic Algorithm
# If we want to use gui we need to put as first parameter 'queue'
def genetic_algorithm(cipher_text, optimization="None", population_size=120, max_mutation_rate=0.4,
                      min_mutation_rate=0.05, max_iterations=1000, elitism=True, fitness_stagnation_threshold=15):
    # Generate initial population of solutions (= permutation of letters)
    population = [init_generate_solution() for _ in range(population_size)]
    best_score = float('-inf')
    max_score = float('-inf')
    perv_score = 0
    best_solution = ''
    stop_counter = 0  # Counter to track the number of iterations with no improvement
    # max_mutation_rate = 0.001
    # min_mutation_rate = 0.3
    # max_iterations = 300
    mutation_rate = 0.1
    # Start with an initial best score
    previous_best_score = float('-inf')
    previous_best_avg = float('-inf')

    # Track the rate of improvement over the last few generations
    improvement_rates = deque(maxlen=5)
    for iteration in range(max_iterations):
        # mutation rate starts at the maximum value and
        # decreases linearly to the minimum value over the course of the iterations.
        # mutation_rate = max_mutation_rate - (max_mutation_rate - min_mutation_rate) * (iteration/max_iterations)
        # I think it maybe a good function to use for darwin:
        # mutation_rate =+ (max_mutation_rate + (iteration/max_iterations))

        #  simulated annealing:
        # mutation_rate = max_mutation_rate * (1 + iteration)

        if optimization == 'None':
            scores, letter_freq_opt = fitness(population,
                                              cipher_text)  # evaluates how good each solution in the population is.
            # scores = [simplified_fitness(solution, cipher_text) for solution in population]
        else:
            if optimization == 'lamarckian':
                print("You are using Lamarckian optimization")

                lamarkian = True
            else:
                # max_mutation_rate = 0.1
                # min_mutation_rate = 0.05
                # max_iterations = 1000
                # mutation_rate = max_mutation_rate + (iteration / max_iterations)
                lamarkian = False
                print("You are using Darwinian optimization")

            # Perform local optimization on each solution before fitness evaluation
            # Initialize an empty list to store the optimized population
            population_optimized = []

            # Perform local optimization on the current solution
            optimized_solutions, optimized_scores, letter_freq_opt = local_optimization(population, cipher_text, N=5,
                                                                                        lamarkian=lamarkian)

            # Add the optimized solution and score to the list
            # population_optimized.append((optimized_solutions, optimized_score))

            # Add the optimized solutions and scores to the list
            population_optimized.extend(zip(optimized_solutions, optimized_scores))

            # Separate the optimized solutions and their scores
            solutions, scores = zip(*population_optimized)

            # Assign the (potentially) optimized solutions back to the population
            population = solutions

        max_score = max(scores)  # find the highest fitness score in the current population.
        max_index = scores.index(max_score)  # find the index of the solution that achieved this score

        min_score = min(scores)
        min_index = scores.index(min_score)
        # if max_score == best_score:  # no improvement
        #     mutation_rate += 0.05  # increase mutation rate
        # else:
        #     mutation_rate -= 0.05  # decrease mutation rate
        #     mutation_rate = max(min_mutation_rate, mutation_rate)  # ensure mutation rate doesn't fall below minimum
        #######################################################################
        # Calculate the rate of improvement
        sumScore = 0
        for score in scores:
            sumScore = + score
        avg_score = sumScore / len(scores)

        #improvment_avg_score = abs(avg_score-previous_best_avg)
        improvement_rate = max_score - previous_best_score
        improvement_rates.append(improvement_rate)
        #improvement_rates.append(improvment_avg_score)
        # Calculate the average rate of improvement
        average_improvement_rate = sum(improvement_rates) / len(improvement_rates)
        print(f"Average improvement_rate: {average_improvement_rate}")

        # Calculate the difference between the average improvement rate and the threshold
        difference_from_threshold = abs(average_improvement_rate - 0.005)

        # If the rate of improvement is lower than a threshold (here 0.01 is an example value),
        # increase the mutation rate
        if average_improvement_rate <= 0.005:
            #mutation_rate += 0.2 * difference_from_threshold  # change is proportional to difference from threshold
            mutation_rate += 0.08

            mutation_rate = min(max_mutation_rate, mutation_rate)  # ensure mutation rate doesn't exceed maximum

        # If the rate of improvement is higher than the threshold, decrease the mutation rate
        else:
            #mutation_rate -= 0.1 * difference_from_threshold  # change is proportional to difference from threshold
            mutation_rate -= 0.05
            mutation_rate = max(min_mutation_rate, mutation_rate)  # ensure mutation rate doesn't fall below minimum


        previous_best_score = max_score
        #previous_best_avg = avg_score

        #########################################################################
        if max_score > best_score:
            best_score = max_score
            best_solution = population[max_index]
            stop_counter = 0  # Reset the counter if there is an improvement
        else:
            stop_counter += 1  # Increment the counter if there is no improvement
        print(
            f"Iteration: {iteration}, Best solution: {best_solution}, Fitness: {best_score}, Mutation rate: {mutation_rate}")
        # queue.put(f"Iteration: {iteration}, Best solution: {best_solution}, Fitness: {best_score:.3f}, Mutation rate: {mutation_rate:.3f}")

        if stop_counter >= fitness_stagnation_threshold:
            print(
                f"No improvement in fitness score for {fitness_stagnation_threshold} iterations. Stopping the algorithm.")
            # queue.put(f"No improvement in fitness score for {fitness_stagnation_threshold} iterations. Stopping the algorithm.")

            break

        selected = selection(population, scores)

        population = generate_new_solutions(selected, mutation_rate, letter_freq_opt)

        # Elitism - Ensure the best solution is always in population
        if elitism and best_solution not in population:
            # If the best solution is not already in the population, it replaces a random solution.
            population[random.randint(0, population_size - 1)] = best_solution
        if elitism and min_score in population:
            population[min_index] = best_solution

    # After all iterations, decipher the text with best solution
    deciphered_text = decipher_convert_format(cipher_text, best_solution)

    # Write the deciphered text into plain.txt
    with open('plain.txt', 'w') as f:
        f.write(deciphered_text)

    # Write the permutation table into perm.txt
    with open('perm.txt', 'w') as f:
        for i in range(26):
            ###################### need to change to best_solution[i].upper() ##################################################
            f.write(f"{string.ascii_lowercase[i]} {best_solution[i]}\n")

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
    with open('test1enc.txt', 'r') as f:
        cipher_text = f.read().strip()

    genetic_algorithm(cipher_text, optimization='None', population_size=200, max_mutation_rate=0.4, min_mutation_rate=0.02,
                      max_iterations=300, elitism=True)
    #true_coding_file = 'true_perm.txt'  # Replace with the actual file name and path
    #results_file = 'perm.txt'  # Replace with the actual file name and path

    #accuracy = calculate_accuracy(true_coding_file, results_file)
    #print(f"Accuracy: {accuracy:.2f}%")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    exit(1)
