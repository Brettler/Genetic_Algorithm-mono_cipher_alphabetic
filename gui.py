import tkinter as tk
import GeneticAlgorithm
from tkinter import ttk
import threading
import queue
import sys
message_queue = []  # Store the messages
current_message_index = 0  # Index of the current message being displayed
"""
def check_queue(q):
    try:
        msg = q.get_nowait()  # try to get a message from the queue
    except queue.Empty:
        pass  # if there's nothing in the queue, do nothing
    else:
        output_text.insert(tk.END, msg + '\n')  # add the message to the text widget
        output_text.see(tk.END)  # scroll to the last line

    root.after(1, check_queue, q)  # check the queue again after 100 ms
"""

def check_queue(q):
    global current_message_index

    try:
        msg = q.get_nowait()  # try to get a message from the queue
    except queue.Empty:
        pass  # if there's nothing in the queue, do nothing
    else:
        message_queue.append(msg)  # add the message to the queue

    if current_message_index < len(message_queue):
        output_text.insert(tk.END, message_queue[current_message_index] + '\n')  # display the current message
        output_text.see(tk.END)  # scroll to the last line
        current_message_index += 1

    if current_message_index >= len(message_queue):
        root.after(100, check_queue, q)  # wait for 5 seconds before checking the queue again
    else:
        root.after(100, check_queue, q)  # check the queue again after 1 ms


def on_close():
    print("Closing program...")
    root.destroy()
    sys.exit()

def run_genetic_algorithm():
    cipher_text_path = 'enc.txt'
    with open(cipher_text_path, 'r') as f:
        cipher_text = f.read().strip()


    population_size = int(population_size_entry.get())
    max_mutation_rate = float(max_mutation_rate_entry.get())
    min_mutation_rate = float(min_mutation_rate_entry.get())
    max_iterations = int(max_iterations_entry.get())
    elitism = bool(int(elitism_entry.get()))
    optimization = optimization_combobox.get()  # get the selected value from the combobox

    #GeneticAlgorithm.genetic_algorithm(cipher_text, optimization, population_size, max_mutation_rate, min_mutation_rate, max_iterations, elitism)
    q = queue.Queue()
    thread = threading.Thread(target=GeneticAlgorithm.genetic_algorithm, args=(q, cipher_text, optimization, population_size, max_mutation_rate, min_mutation_rate, max_iterations, elitism))
    thread.daemon = True
    thread.start()
    check_queue(q)  # check the queue for messages right away

root = tk.Tk()


population_size_label = tk.Label(root, text="Population number:")
population_size_label.pack()
population_size_entry = tk.Entry(root)
population_size_entry.pack()
population_size_entry.insert(0, "120")

max_mutation_rate_label = tk.Label(root, text="Mutation rate:")
max_mutation_rate_label.pack()
max_mutation_rate_entry = tk.Entry(root)
max_mutation_rate_entry.pack()
max_mutation_rate_entry.insert(0, "0.4")

min_mutation_rate_label = tk.Label(root, text="Minimum mutation rate:")
min_mutation_rate_label.pack()
min_mutation_rate_entry = tk.Entry(root)
min_mutation_rate_entry.pack()
min_mutation_rate_entry.insert(0, "0.05")

max_iterations_label = tk.Label(root, text="Maximum iterations:")
max_iterations_label.pack()
max_iterations_entry = tk.Entry(root)
max_iterations_entry.pack()
max_iterations_entry.insert(0, "1000")

elitism_label = tk.Label(root, text="Elitism:")
elitism_label.pack()
elitism_entry = tk.Entry(root)
elitism_entry.pack()
elitism_entry.insert(0, "1")

# Create the combobox for selecting the optimization strategy
optimization_label = tk.Label(root, text="Optimization strategy:")
optimization_label.pack()
optimization_combobox = ttk.Combobox(root, values=["None", "Darwinian", "Lamarckian"])
optimization_combobox.pack()
optimization_combobox.current(0)  # set initial selection to "None"

run_button = tk.Button(root, text="Run", command=run_genetic_algorithm)
run_button.pack()

# We choose the width by the amount of characters.
output_text = tk.Text(root, width=100)
output_text.pack()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()