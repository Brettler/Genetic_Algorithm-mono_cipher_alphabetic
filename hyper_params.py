# Define the shared variables with their default values.
class hyper_parameter_object:
    def __init__(self):
        self.word_hyper_param = 1
        self.letter_hyper_param = 0.5
        self.pair_letters_hyper_param = 0.5
        self.hyper_letter_correct = 0.5
        self.hyper_pair_letters_correct = 0.5
        self.mutation_trashold = 0.005
        self.increase_mutation = 0.08
        self.decrease_mutation = 0.05
        self.improvement_rates_queue_length = 5
        self.N = 5
        self.random_mutation_func = False
        self.input_enc_file = 'enc.txt'
        self.true_perm_file = 'true_perm.txt'
