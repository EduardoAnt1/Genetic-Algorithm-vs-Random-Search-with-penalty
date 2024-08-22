import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time
import pandas as pd

# Função do desafio 14
def objective_function(args):
    f = 0
    x = args
    tam = x.shape[0]
    c = [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708, -26.662, -22.179]
    for i in range(tam):
        if x[i] <= 0 or sum(x) == 0 or x[i] > 10:
            return np.inf  # Penalizar com infinito se houver um valor inválido
        f += x[i] * (c[i] + np.log(x[i] / sum(x)))
    return f

# Função de restrições
def function_constraints(args):
    x = args
    h = [
        x[0] + 2 * x[1] + 2 * x[2] + x[5] + x[9] - 2,
        x[3] + 2 * x[4] + x[5] + x[6] - 1,
        x[2] + x[6] + x[7] + 2 * x[8] + x[9] - 1
    ]
    return h

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
        if i + 1 < num_parents and np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, num_variables)
            children[i] = np.concatenate((parents[i][:crossover_point], parents[i+1][crossover_point:]))
            children[i+1] = np.concatenate((parents[i+1][:crossover_point], parents[i][crossover_point:]))
        else:
            children[i] = parents[i]
        if i + 1 < num_parents:
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
    return np.apply_along_axis(objective_function, 1, population)

# Função para avaliar o fitness de cada indivíduo na população, com penalização por barreira
def evaluate_population_with_barrier(population, constraint_func):
    fitness = evaluate_population(population)
    feasible_solutions = []
    for i, sol in enumerate(population):
        constraints = constraint_func(sol)
        if all(c <= 0.001 for c in constraints):
            feasible_solutions.append(i)  # Armazena índices de soluções viáveis
        else:
            fitness[i] = np.inf  # Penalizar soluções inviáveis com um fitness infinito
    # Se não houver soluções viáveis, permita apenas o melhor resultado não viável
    if not feasible_solutions:
        min_fitness_idx = np.argmin(fitness)
        fitness[min_fitness_idx] = np.inf  # Penaliza a solução não viável com fitness infinito
    return fitness

# Função para avaliar o fitness de cada indivíduo na população, com penalização dinâmica
def evaluate_population_with_dynamic_penalty(population, constraint_func):
    fitness = np.empty(population.shape[0])
    best_feasible_fitness = np.inf  # Inicializa com infinito, indicando que nenhuma solução viável foi encontrada
    found_feasible_solution = False

    for i, sol in enumerate(population):
        constraints = constraint_func(sol)
        if all(c <= 0.001 for c in constraints):
            current_fitness = objective_function(sol)
            fitness[i] = current_fitness
            if current_fitness < best_feasible_fitness:
                best_feasible_fitness = current_fitness
                found_feasible_solution = True
        else:
            # Se uma solução viável não foi encontrada, aplica uma penalidade
            penalty = sum(max(0, -a) for a in constraints)
            fitness[i] = objective_function(sol) + abs(penalty) + 2200
            # Se uma solução viável já foi encontrada, não permitir que soluções inviáveis sejam melhores
            if found_feasible_solution:
                fitness[i] = max(fitness[i], best_feasible_fitness)
    return fitness

# Função para encontrar o melhor indivíduo na população
def find_best_individual(population, fitness):
    best_idx = np.argmin(fitness)
    return population[best_idx], fitness[best_idx]

# Algoritmo Genético
def genetic_algorithm(num_generations, pop_size, num_variables, min_val, max_val, crossover_rate, mutation_rate, tournament_size, penalty_type='barrier'):
    population = initialize_population(pop_size, num_variables, min_val, max_val)
    best_fitness = []
    best_solutions = []
    all_solutions = []
    all_fitness = []
    
    for _ in range(num_generations):
        if penalty_type == 'barrier':
            fitness = evaluate_population_with_barrier(population, function_constraints)
        elif penalty_type == 'dynamic':
            fitness = evaluate_population_with_dynamic_penalty(population, function_constraints)
        else:
            raise ValueError("Invalid penalty type. Choose 'barrier' or 'dynamic'.")

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
num_generations = [100,1000,10000]
num_variables = 10
pop_size = 50
min_val = [0] * num_variables
max_val = [10] * num_variables
crossover_rate = 0.8
mutation_rate = 0.1
tournament_size = 5

# Pasta para salvar as imagens
output_dir = "genetic_algorithm_results"
os.makedirs(output_dir, exist_ok=True)

# Lista para armazenar resultados
results = []

# Execução do algoritmo genético com ambas as penalizações
penalty_types = ['barrier', 'dynamic']
for penalty_type in penalty_types:
    feasible_solutions_count = 0
    for num_g in num_generations:
        fitness_results = []
        execution_times = []

        for i in range(25):
            np.random.seed(i)
            start_time = time.time()
            if penalty_type == 'barrier':
                best_fitness, best_solutions, all_fitness, all_solutions = genetic_algorithm(
                    num_g, pop_size, num_variables, min_val, max_val, crossover_rate, mutation_rate, tournament_size, penalty_type='barrier'
                )
            elif penalty_type == 'dynamic':
                best_fitness, best_solutions, all_fitness, all_solutions = genetic_algorithm(
                    num_g, pop_size, num_variables, min_val, max_val, crossover_rate, mutation_rate, tournament_size, penalty_type='dynamic'
                )
            else:
                raise ValueError("Invalid penalty type. Choose 'barrier' or 'dynamic'.")

            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            fitness_results.append(best_fitness)
            # Gráficos de convergência
            plt.figure(figsize=(10, 6))
            num_iterations = pop_size * num_g
            for i, fitness in enumerate(fitness_results):
                # Substitui np.inf por um valor grande para visualização
                y_values = [val if val != np.inf else 1e10 for val in fitness]
                x_values = np.arange(len(y_values)) * pop_size # Ajusta o eixo x

            # Adiciona a linha horizontal somente para penalização por barreira
            if penalty_type == 'barrier':
                plt.axhline(y=1e10, color='r', linestyle='--', label='Penalização Infinita')

            plt.xlabel('Iterações')
            plt.ylabel('Fitness')
            plt.title(f'Convergência - Penalização {penalty_type.capitalize()}')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{output_dir}/convergence_{penalty_type}_{num_g}.png")
            plt.close()



        # Salvando os resultados
        avg_fitness = np.mean([f[-1] for f in fitness_results])
        best_avg_fitness = np.mean([f[0] for f in fitness_results])
        avg_time = np.mean(execution_times)
        results.append([penalty_type, num_g, avg_fitness, best_avg_fitness, avg_time, feasible_solutions_count])

# Criando o DataFrame de resultados
df_results = pd.DataFrame(results, columns=['Penalização', 'Gerações', 'Fitness Médio', 'Melhor Fitness', 'Tempo Médio', 'Soluções Viáveis'])
df_results.to_excel(f"{output_dir}/results.xlsx", index=False)
