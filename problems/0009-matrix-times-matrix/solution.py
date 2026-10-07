def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    a_m : int = len(a)
    a_n : int = len(a[0])

    b_n : int = len(b[0])
    b_p : int = len(b)

    if a_n != b_p:
        return -1

    c : list[list[int|float]] = [[0 for _ in range(b_p)] for _ in range(a_m)]

    for i in range(a_m):
        for j in range(b_p):
            c[i][j] = sum([a[i][k] * b[k][j] for k in range(b_p)])
    
    return c