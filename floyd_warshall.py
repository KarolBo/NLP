from math import inf

weights = [[0, 1, 2, inf, inf, inf],
           [inf, 0, inf, 1, 5, inf],
           [inf, 2, 1, 2, 3, inf],
           [4, inf, inf, 1, inf, 4],
           [inf, inf, inf, inf, 10, 2],
           [inf, inf, inf, 3, inf, 0]]

pointers = [['a', 'b', 'c', None, None, None],
            [None, 'b', None, 'd', 'e', None],
            [None, 'b', 'c', 'd', 'e', None],
            ['a', None, None, 'd', None, 'f'],
            [None, None, None, None, 'e', 'f'],
            [None, None, None, 'd', None, 'f']]

n = len(weights)

for k in range(n):
    print('Iteration:', k)
    for row in weights:
        print(row)
    print()
    for row in pointers:
        print(row)
    print()
    for i in range(n):
        for j in range(n):
            if weights[i][j] > weights[i][k] + weights[k][j]:
                weights[i][j] = weights[i][k] + weights[k][j]
                pointers[i][j] = pointers[i][k]


def get_path(u, v):
    nodes = ['a', 'b', 'c', 'd', 'e', 'f']
    u = nodes.index(u)
    v = nodes.index(v)
    print(u, v)
    if not pointers[u][v]:
        return None
    path = [nodes[u]]
    while u != v:
        path.append(pointers[u][v])
        u = nodes.index(pointers[u][v])
    return path


print(get_path('f', 'b'))

