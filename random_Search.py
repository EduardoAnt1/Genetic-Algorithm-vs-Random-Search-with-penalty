# %%
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os

# Função drange
def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

# Função objetivo
def funcao_objetivo(tam, x, c):
    f = 0
    for i in range(tam):
        f += x[i] * (c[i] + np.log(x[i] / sum(x)))
    return f

# Função de verificação de viabilidade
def isFeasible(x):
    h = [None] * 3
    h[0] = x[0] + 2 * x[1] + 2 * x[2] + x[5] + x[9] - 2
    h[1] = x[3] + 2 * x[4] + x[5] + x[6] - 1
    h[2] = x[2] + x[6] + x[7] + 2 * x[8] + x[9] - 1
    return h

# Função de busca aleatória
def busca_aleatoria(qtd_interacoes, limite_inf, limite_sup, tam):
    c = [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708, -26.662, -22.179]
    x = [None] * tam
    melhor_x = [None] * tam
    melhor_y = np.inf
    melhores_imagens = []
    tem_factivel = False

    for p in range(qtd_interacoes):
        for i in range(tam):
            x[i] = random.uniform(limite_inf, limite_sup)
        y = funcao_objetivo(tam, x, c)
        if tem_factivel:
            if melhor_y > y and all(i <= 0.001 for i in isFeasible(x)):
                if melhor_y > y:
                    melhor_y = y
                    melhor_x = x.copy()
        else:
            if melhor_y > y and all(i <= 0.001 for i in isFeasible(x)):
                if melhor_y > y:
                    melhor_y = y
                    melhor_x = x.copy()
                    tem_factivel = True
            if not all(i <= 0.001 for i in isFeasible(x)):
                y += np.sum(np.abs(isFeasible(x))) + 2200
                if melhor_y > y:
                    melhor_y = y
                    melhor_x = x.copy()
        melhores_imagens.append(melhor_y)
    return melhor_x, melhor_y, melhores_imagens

# Configuração dos diretórios e execução
qtd = 500
output_dir = r'C:\Users\eduar\OneDrive\Área de Trabalho\graficos'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for k in range(3):
    factivel = 0
    dados = []
    date = []
    tempo = []
    tabela = []
    qtd = qtd * 10
    print('Número de interações:', qtd, '\n')

    for i in range(25):
        random.seed(i)
        inicio = time.time()
        x, y, melhor_y = busca_aleatoria(qtd, 0, 10, 10)
        fim = time.time()
        tempo.append(fim - inicio)
        if all(i <= 0.001 for i in isFeasible(x)):
            factivel += 1.0
        dados.append(melhor_y)
        date.append(y)
        plt.title('Convergência')
        plt.xlabel('Iterações')
        plt.ylabel('Melhor valor')
        plt.axvline(0, color='k')
        plt.axhline(0, color='k')
        plt.plot(range(len(melhor_y)), melhor_y, marker=',')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'convergence_{k+1}_{i}.png'))
        plt.close()
    plt.title('Convergência')
    plt.xlabel('Iterações')
    plt.ylabel('Melhor valor')
    plt.axvline(0, color='k')
    plt.axhline(0, color='k')
    for j in range(10):
        plt.plot(range(len(dados[j])), dados[j], marker=',')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'convergence_{k+1}.png'))
    plt.show()
    plt.close()

    tabela = {
        'Variáveis': ['Melhor', 'Pior', 'Média', 'Mediana', 'Desvio Padrão', 'Factível por Execução', 'Tempo médio de execução'],
        'Resultados': [min(date), max(date), np.mean(date), np.median(date), np.std(date), factivel, np.mean(tempo)]
    }
    df = pd.DataFrame(tabela)

    # Salvando o DataFrame em um arquivo XLSX
    try:
        df.to_excel(os.path.join(output_dir, f'resultados_{k+1}.xlsx'), index=False)
        print(f'Arquivo resultados_{k+1}.xlsx salvo com sucesso!')
    except Exception as e:
        print(f'Erro ao salvar o arquivo XLSX: {e}')
