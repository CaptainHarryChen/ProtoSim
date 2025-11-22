import sympy as sp

def create_vector(name, size=3):
    return sp.Matrix(sp.symbols(f'{name}[0] {name}[1] {name}[2]', real=True))

def create_matrix(name, rows=3, cols=3):
    symbols_list = []
    for i in range(rows):
        for j in range(cols):
            symbols_list.append(f'{name}[{i}*{cols}+{j}]')
    return sp.Matrix(sp.symbols(' '.join(symbols_list), real=True)).reshape(rows, cols)
