import sympy as sp
from sympy import symbols, diff, ccode, det
from utils import create_matrix

def flatten_X(code):
    for i in range(4):
        for j in range(4):
            code = code.replace(f'X[{i}*4+{j}]', f'X[{i * 4 + j}]')
    return code

X = create_matrix('X', 4, 4, flatten=False)
J = det(X)
X_inv = X.inv()
X_inv_mul_J = X_inv * J

print('// Inverse of 4x4 matrix multiplied by its determinant')
for i in range(4):
    for j in range(4):
        print(f'inv_X[{i * 4 + j}] = {flatten_X(ccode(X_inv_mul_J[i, j]))};')
print(f'J = {flatten_X(ccode(J))};')
