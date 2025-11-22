import sympy as sp
from sympy import symbols, diff, ccode, log, det, simplify, cse
from utils import create_vector, create_matrix

x0, x1, x2, x3 = [create_vector(f'x{i}') for i in range(4)]
InvDm = create_matrix('InvDm')
W, mu, lambda_ = symbols('W mu lambda', real=True)

Ds = sp.Matrix(3, 3, lambda i, j: [x1, x2, x3][j][i] - x0[i])

F = Ds * InvDm
F_inv = F.inv()
F_inv_T = F_inv.T
P = mu * (F - F_inv_T) + lambda_ * log(det(F)) * F_inv_T
H = -W * P * InvDm.T
f0 = sp.Matrix(3, 1, lambda i, j: - (H[i, 0] + H[i, 1] + H[i, 2]))

K = []

for i in range(12):
    j = i // 3
    k = i % 3
    if i < 3:
        K.append(diff(f0[k], x0[k]))
    else:
        K.append(diff(H[k, j - 1], [x1, x2, x3][j - 1][k]))

with open('build/output.txt', 'w') as f:
    sub_expr, simplified_expr = cse(K, symbols=sp.numbered_symbols(prefix='t'))
    for expr in sub_expr:
        f.write(f'{expr[0]} = {ccode(expr[1])};\n')
    for i in range(len(simplified_expr)):
        f.write(f'K[{i}] = {ccode(simplified_expr[i])};\n')
