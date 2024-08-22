import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time

# Função do desafio 14
def objetive_function(args):
    f = 0
    x = args
    tam = x.shape[0]
    c = [-6.089,-17.164,-34.054,-5.914,-24.721,-14.986,-24.1,-10.708,-26.662,-22.179]
    for i in range(tam):
        f += x[i] * (c[i] + np.log(x[i] / sum(x)))
    return f

# Função de restrições
def function_constraints(args):
    x = args
    h = [None]*3
    h[0] = x[0] + 2*x[1] + 2*x[2] + x[5] + x[9] - 2
    h[1] = x[3] + 2*x[4] + x[5] + x[6] - 1
    h[2] = x[2] + x[6] + x[7] + 2*x[8] + x[9] - 1
    return h

fitness_function = objetive_function

constraint_func= function_constraints

# Função para inicializar a população
def initialize_population(pop_size, num_variables, min_val, max_val):
    return np.random.uniform(min_val, max_val, (pop_size, num_variables))

# Função de seleção de pais (Torneio)
def select_parents(population, fitness, tournament_size):
    selected_parents = []
    pop_size = population.shape[0]
    for _ in range(pop_size):
        tournament_indices = np.random.choice(range(pop_size), size=tournament_size, replace=False)
        tournament_fitness = fitness[tournament_indices]
        selected_parent_idx = tournament_indices[np.argmin(tournament_fitness)]
        selected_parents.append(population[selected_parent_idx])
    return np.array(selected_parents)

# Função de crossover (Ponto Único)
def crossover(parents, crossover_rate):
    num_parents, num_variables = parents.shape
    children = np.empty((num_parents, num_variables))
    for i in range(0, num_parents, 2):
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, num_variables)
            children[i] = np.concatenate((parents[i][:crossover_point], parents[i+1][crossover_point:]))
            children[i+1] = np.concatenate((parents[i+1][:crossover_point], parents[i][crossover_point:]))
        else:
            children[i] = parents[i]
            children[i+1] = parents[i+1]
    return children

# Função de mutação (Uniforme)
def mutate(children, mutation_rate, min_val, max_val):
    mutated_children = np.copy(children)
    for child in mutated_children:
        for i in range(len(child)):
            if np.random.rand() < mutation_rate:
                child[i] = np.random.uniform(min_val[i], max_val[i])
    return mutated_children

# Função para avaliar o fitness de cada indivíduo na população
def evaluate_population(population):
    return np.apply_along_axis(fitness_function, 1, population)

# Função para avaliar o fitness de cada indivíduo na população, com barreira para soluções inviáveis
def evaluate_population_with_barrier(population, constraint_func):
    fitness = np.apply_along_axis(fitness_function, 1, population)
    for i, sol in enumerate(population):
        if not all(c >= 1*10**-5 for c in constraint_func(sol)):
            fitness[i] = np.inf  # Penalizar soluções inviáveis com um fitness infinito
    return fitness

# Função para encontrar o melhor indivíduo na população
def find_best_individual(population, fitness):
    best_idx = np.argmin(fitness)
    return population[best_idx], fitness[best_idx]

# Algoritmo Genético
def genetic_algorithm(num_generations, pop_size, num_variables, min_val, max_val, crossover_rate, mutation_rate, tournament_size):
    population = initialize_population(pop_size, num_variables, min_val, max_val)
    best_fitness = []
    best_solutions = []
    all_solutions = []
    all_fitness = []
    for _ in range(num_generations):
        fitness = evaluate_population_with_barrier(population, constraint_func)
        best_individual, best_fitness_value = find_best_individual(population, fitness)
        best_fitness.append(best_fitness_value)
        best_solutions.append(best_individual)
        for i in range(pop_size):
            all_solutions.append(population[i])
            all_fitness.append(fitness[i])
        parents = select_parents(population, fitness, tournament_size)
        children = crossover(parents, crossover_rate)
        mutated_children = mutate(children, mutation_rate, min_val, max_val)
        population = mutated_children
    return best_fitness, best_solutions, all_fitness, all_solutions

# Parâmetros do algoritmo genético
num_generations = [100, 1000, 10000]
num_variables = 10
pop_size = 50
min_val = [0]*num_variables
max_val = [10]*num_variables
crossover_rate = 0.8
mutation_rate = 0.1
tournament_size = 5

# Pasta para salvar as imagens
output_dir = "genetic_algorithm_results"
os.makedirs(output_dir, exist_ok=True)

# Lista para armazenar resultados
results = []

# Execução do algoritmo genético
for num_g in num_generations:
    fitness_results = []
    execution_times = []
    feasible_solutions_count = []
    
    for i in range(10):
        random.seed(i)
        start_time = time.time()
        best_fitness, best_solutions, all_fitness, all_solutions = genetic_algorithm(
            num_g, pop_size, num_variables, min_val, max_val, crossover_rate, mutation_rate, tournament_size
        )
        execution_time = time.time() - start_time
        execution_times.append(execution_time)
        fitness_results.append(best_fitness[-1])
        feasible_solutions = sum(1 for f in all_fitness if f < np.inf)
        feasible_solutions_count.append(feasible_solutions)

        # Plotando o gráfico da convergência do algoritmo
        plt.figure(figsize=(10, 5))
        plt.plot(best_fitness, label='Melhor Fitness')
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.title(f'Convergência do Algoritmo Genético - {num_g} Gerações')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/convergence_{num_g}_gen_{i}.png")
        plt.close()

        # Dados para plot do gráfico 3D
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        x1_v = np.array(all_solutions)[:, 0]
        x2_v = np.array(all_solutions)[:, 1]

        surf = ax.scatter(x1_v, x2_v, all_fitness, s=1, color='r')
        plt.savefig(f"{output_dir}/3Dscatter_{num_g}_gen_{i}.png")
        plt.close()

    # Cálculos das métricas
    best_result = np.min(fitness_results)
    worst_result = np.max(fitness_results)
    std_dev = np.std(fitness_results)
    median_result = np.median(fitness_results)
    avg_time = np.mean(execution_times)
    avg_feasible_solutions = np.mean(feasible_solutions_count)

    # Adicionando resultados à lista
    results.append((num_g, best_result, worst_result, std_dev, median_result, avg_time, avg_feasible_solutions))

# Criando a tabela de resultados
import pandas as pd

df_results = pd.DataFrame(results, columns=[
    "Gerações", "Melhor Fitness", "Pior Fitness", "Desvio Padrão", "Mediana", "Tempo Médio (s)", "Soluções Factíveis"
])

# Salvando a tabela como imagem
df_results.to_csv(f"{output_dir}/genetic_algorithm_summary.csv", index=False)
