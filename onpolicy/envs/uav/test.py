C = [[9.34e-01, 1.97e-02, -1.24e-04, 2.73e-07], [2.30e-01, 2.44e-03, -3.34e-06], [-2.25e-03, 6.58e-06], [1.86e-05]]
alpha, beta, gamma = 0.3, 500, 1.5
z = 0
for j in range(0, 4):
    for i in range(0, 4 - j):
        z += C[i][j] * ((alpha*beta)** i) * (gamma ** j)
print(z)