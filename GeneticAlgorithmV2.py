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
def fitness(solutions, cipher_text, hyper_params):
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

        # fitness_score = (word_count / total_count) - 0.5 * letter_freq_fitness - 0.5 * pair_letters_fitness \
        #                 + 0.5*pair_letters_correct/(len(deciphered_text) - 1) + 0.5*letter_correct/len(deciphered_text)

        fitness_score = hyper_params.word_hyper_param * (word_count / total_count) \
                        - hyper_params.letter_hyper_param * letter_freq_fitness\
                        - hyper_params.pair_letters_hyper_param * pair_letters_fitness \
                        + hyper_params.hyper_pair_letters_correct * pair_letters_correct / (len(deciphered_text) - 1)\
                        + hyper_params.hyper_letter_correct * letter_correct / len(deciphered_text)
        # append the fitness_score of this solution to the fitness_scores list
        fitness_scores.append(fitness_score)

    return fitness_scores, letter_freq_dif_list


# Decipher function: replaces each letter in the cipher text with the corresponding letter in the solution.
def decipher(cipher_text, perm_rules_decoder):
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


# PMX crossover function: similar to the above,
# but uses a mapping between parent genes to ensure child genes are not repeated.
def crossover(parent1, parent2):
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
    # Fill the rest of the child solution.
    for i, letter in enumerate(parent2):  # Iterate over the letters of parent2
        # Check if the letter is outside the selected range because in the selected range the child already have
        # the letters from parent1.
        if i not in range(start, end):
            # If the letter we are current iterate is already in the mapping it means this letter is in the parent1.
            # If this letter is in parent1 we wont add it to the child because the child already contain it.
            while letter in mapping:  # Check if the letter is already in the mapping
                # Replace the letter with the corresponding letter from parent2 based on the mapping
                letter = mapping[letter]
            child[i] = letter  # Assign the letter to the child
    # Return the child as a string
    return ''.join(child)


# Mutation function: performs a swap mutation on a solution if a random number is less than the mutation rate.
def random_mutation(solution, mutation_rate):
    # Swap mutation only if mutation rate condition is satisfied
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(solution)), 2)
        mutated = list(solution)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return ''.join(mutated)
    return solution


def mutation(solution, mutation_rate, max_letter, min_letter):
    # Swap mutation only if mutation rate condition is satisfied
    if random.random() < mutation_rate:
        i = solution.index(max_letter)  # find the index of max_letter
        idx_min = solution.index(min_letter)  # find the index of min_letter
        # choose a random index different from i and not equal to the index of min_letter
        # inx_min describe the letter with good frequency , i describe a letter with bad frequency.
        # The goal is to try to replace the bad letter (i) with random letter that it is not the best letter (idx_min)
        j = random.choice([k for k in range(len(solution)) if k != i and k != idx_min])
        mutated = list(solution)
        mutated[i], mutated[j] = mutated[j], mutated[i]  # swap letters at positions i and j
        return ''.join(mutated)
    return solution


# Generate new solutions for the next generation by applying crossover and mutation to the selected population.
def generate_new_solutions(selected_population, mutation_rate, letter_freq_opt, random_mutation_func):
    population_size = len(selected_population)
    new_population = []

    for i in range(population_size // 2):
        parent1, parent2 = random.sample(selected_population, 2)

        child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)

        parent1_idx = selected_population.index(parent1)
        parent2_idx = selected_population.index(parent2)
        max_letter_daddy = max(letter_freq_opt[parent1_idx].items(), key=lambda x: x[1])[0]
        min_letter_daddy = min(letter_freq_opt[parent1_idx].items(), key=lambda x: x[1])[0]

        max_letter_mommy = max(letter_freq_opt[parent2_idx].items(), key=lambda x: x[1])[0]
        min_letter_mommy = min(letter_freq_opt[parent1_idx].items(), key=lambda x: x[1])[0]

        child1 = mutation(child1, mutation_rate, max_letter_daddy, min_letter_daddy)
        child2 = mutation(child2, mutation_rate, max_letter_mommy, min_letter_mommy)
        if random_mutation_func:
            child1 = random_mutation(child1, mutation_rate)
            child2 = random_mutation(child2, mutation_rate)
        new_population.extend([child1, child2])
    return new_population


def local_optimization(population, cipher_text, hyper_params, lamarkian=False):
    # genome = solutions (=permutations)
    # nucleotide = letter in the solution
    # Initialize the best solutions and scores with the current population
    original_solutions = list(population)
    modify_genome = list(population)
    # Calculate the fitness scores for the entire population
    original_best_scores, original_letter_freq = fitness(population, cipher_text, hyper_params)

    for idx, candidate_dna_string in enumerate(modify_genome):
        candidate_idx = original_solutions.index(candidate_dna_string)
        # Sorting the frequency dictionary and extracting top 2N frequent letters
        sorted_freq = sorted(original_letter_freq[candidate_idx].items(), key=lambda x: x[1], reverse=True)[
                      :2 * hyper_params.N]
        top_frequent_letters = [letter for letter, freq in sorted_freq]
        # Perform local optimization for the specified number of iterations (N)
        for _ in range(hyper_params.N):
            i, j = random.sample(top_frequent_letters, 2)  # Select two random letters among top frequent ones
            nucleotide = list(candidate_dna_string)
            # Get the indices of the selected letters
            idx1, idx2 = nucleotide.index(i), nucleotide.index(j)
            nucleotide[idx1], nucleotide[idx2] = nucleotide[idx2], nucleotide[idx1]  # Swap the selected letters
            candidate_dna_string = ''.join(nucleotide)
        modify_genome[idx] = candidate_dna_string  # Save the changes back to the genome

    scores_candidates_genome, freq_letter_candidates_genome = fitness(modify_genome, cipher_text, hyper_params)

    for i in range(len(population)):
        # Update the best score and solution if the candidate score is higher
        if scores_candidates_genome[i] > original_best_scores[i]:
            original_best_scores[i] = scores_candidates_genome[i]
            original_letter_freq[i] = freq_letter_candidates_genome[i]

            if lamarkian:
                original_solutions[i] = modify_genome[i]
                original_letter_freq[i] = freq_letter_candidates_genome[i]

    return original_solutions, original_best_scores, original_letter_freq



# Genetic Algorithm
# If we want to use gui we need to put as first parameter 'queue'
def genetic_algorithm(q, hyper_params, fig, canvas, cipher_text,
                      optimization="None",
                      population_size=120,
                      mutation_rate=0.1,
                      max_mutation_rate=0.4,
                      min_mutation_rate=0.1,
                      max_iterations=150,
                      elitism=True,
                      fitness_stagnation_threshold=15):
    q.put(f"You are using the following parameters:"
          f"population_size={population_size}\nmutation_rate={mutation_rate}\nmax_mutation_rate={max_mutation_rate}\nmin_mutation_rate={min_mutation_rate}\nN={hyper_params.N}\n")
    ax = fig.add_subplot(111)
    # Generate initial population of solutions (= permutation of letters)
    population = [init_generate_solution() for _ in range(population_size)]
    best_score = float('-inf')
    min_score = float('-inf')
    best_solution = ''
    stop_counter = 0  # Counter to track the number of iterations with no improvement
    # max_mutation_rate = 0.001
    # min_mutation_rate = 0.3
    # max_iterations = 300
    # mutation_rate = 0.1
    iteration = 0
    # Prepare lists to store iteration and score values
    best_scores = []
    min_scores = []
    iterations = []
    # Start with an initial best score
    previous_best_score = float('-inf')

    if optimization == 'Lamarckian':
        q.put("You are using Lamarckian optimization")
    elif optimization == 'Darwinian':
        q.put("You are using Darwinian optimization")

    # Track the rate of improvement over the last few generations
    improvement_rates = deque(maxlen=hyper_params.improvement_rates_queue_length)

    for iteration in range(max_iterations):

        # Store values to build the graph over each iteration.
        iterations.append(iteration)
        best_scores.append(best_score)
        min_scores.append(min_score)

        ax.clear()  # clear the plot for the new plot
        ax.plot(iterations, best_scores, 'b-', label='Best Score')  # plot best score with blue line
        ax.plot(iterations, min_scores, 'r-', label='Min Score')  # plot worst score with red line

        ax.set_xlabel('Generations')  # Set the x-axis label
        ax.set_ylabel('Fittness Score')  # Set the y-axis label
        ax.legend()  # Display the legend

        if optimization == 'None':
            # evaluates how good each solution in the population is.
            scores, letter_freq_opt = fitness(population, cipher_text, hyper_params)
            ax.set_title(f'Genetic Algorithm - Classic')
        else:
            if optimization == 'Lamarckian':
                lamarkian = True
                ax.set_title(f'Genetic Algorithm - Lamarckian')

            else:
                lamarkian = False
                ax.set_title(f'Genetic Algorithm - Darwinian')

            # Perform local optimization on the current solution
            population, scores, letter_freq_opt = local_optimization(population, cipher_text, hyper_params,
                                                                     lamarkian=lamarkian)

        canvas.draw()  # refresh the canvas

        max_score = max(scores)  # find the highest fitness score in the current population.
        max_index = scores.index(max_score)  # find the index of the solution that achieved this score

        min_score = min(scores)
        min_index = scores.index(min_score)

        # Calculate the rate of improvement
        improvement_rate = max_score - previous_best_score
        improvement_rates.append(improvement_rate)

        # Calculate the average rate of improvement
        average_improvement_rate = sum(improvement_rates) / len(improvement_rates)

        # If the rate of improvement is lower than a threshold (here 0.01 is an example value),
        # increase the mutation rate
        if average_improvement_rate <= hyper_params.mutation_trashold:
            mutation_rate += hyper_params.increase_mutation
            mutation_rate = min(max_mutation_rate, mutation_rate)  # ensure mutation rate doesn't exceed maximum
        # If the rate of improvement is higher than the threshold, decrease the mutation rate
        else:
            mutation_rate -= hyper_params.decrease_mutation
            mutation_rate = max(min_mutation_rate, mutation_rate)  # ensure mutation rate doesn't fall below minimum

        previous_best_score = max_score

        if max_score > best_score:
            best_score = max_score
            best_solution = population[max_index]
            stop_counter = 0  # Reset the counter if there is an improvement
        else:
            stop_counter += 1  # Increment the counter if there is no improvement
        print(
            f"Iteration: {iteration}, Best solution: {best_solution}, Fitness: {best_score}, Average improvement_rate: {average_improvement_rate}, Mutation rate: {mutation_rate}")
        q.put(
            f"Iteration: {iteration}, Best solution: {best_solution}, Fitness: {best_score:.3f}, Average improvement_rate: {average_improvement_rate:.3f}, Mutation rate: {mutation_rate:.3f}")

        if stop_counter >= fitness_stagnation_threshold:
            print(
                f"No improvement in fitness score for {fitness_stagnation_threshold} iterations. Stopping the algorithm.")
            q.put(
                f"No improvement in fitness score for {fitness_stagnation_threshold} iterations. Stopping the algorithm.")

            break

        selected = selection(population, scores)
        population = generate_new_solutions(selected, mutation_rate, letter_freq_opt, hyper_params.random_mutation_func)

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

    true_coding_file = 'true_perm.txt'  # Replace with the actual file name and path
    results_file = 'perm.txt'  # Replace with the actual file name and path

    accuracy = calculate_accuracy(true_coding_file, results_file)

    print(f"Accuracy: {accuracy:.2f}%")
    q.put(f"The algorithm successfully decipher {accuracy:.2f}% correct.")

    random_number = random.random()
    name_graph = f'GeneticGraph_Accuracy={accuracy:.2f}%_{random_number}.png'
    fig.savefig(name_graph)

    if optimization == 'Lamarckian' or optimization == 'Darwinian':
        fitness_calls = 2 * iteration
    else:
        fitness_calls = iteration

    # Set the results into variables.
    number_generations = iteration
    final_solutions = best_solution
    final_fitness_score = best_score
    # We send the last results to the gui so it will be shown to the client when the algorithm stopped.
    q.put(('result', fitness_calls, number_generations, final_fitness_score, final_solutions, accuracy))

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
    pass
    """
    If you want to use the algorithm through this file you will need to change the varibles that genetic_algorithm 
    receive.
    """
    # with open('test1enc.txt', 'r') as f:
    #     cipher_text = f.read().strip()
    # genetic_algorithm(cipher_text, optimization='None', population_size=200, max_mutation_rate=0.4,
    #                   min_mutation_rate=0.02,
    #                   max_iterations=300, elitism=True)


if __name__ == '__main__':
    main()
    exit(1)
