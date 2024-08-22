import numpy as np

# Função do desafio 14
def objective_function(args):
    f = 0
    x = args
    tam = x.shape[0]
    c = [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708, -26.662, -22.179]
    for i in range(tam):
        if x[i] <= 0 or sum(x) == 0:
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

# Vetor da melhor solução existente
best_solution_vector = np.array([0.0406684113216282, 0.147721240492452, 0.783205732104114, 0.00141433931889084, 0.485293636780388, 0.000693183051556082, 0.0274052040687766, 0.0179509660214818, 0.0373268186859717, 0.0968844604336845])

# Avaliar a melhor solução com a função objetivo
best_solution_objective_value = objective_function(best_solution_vector)
print(f"Valor da função objetivo para a melhor solução: {best_solution_objective_value}")

# Avaliar a melhor solução com as restrições
constraints = function_constraints(best_solution_vector)
print(f"Restrições para a melhor solução: {constraints}")
