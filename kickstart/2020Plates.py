def dpsolve(N, P, K, values):
    dp = [0] * (P + 1)
    for i in range(N):
        for p in range(P, 0, -1):
            for j in range(min(p, K)):
                dp[p] = max(dp[p], dp[p - j - 1] + values[i][j])
    return dp[-1]


def solve(N, P, K, values):
    maxvalues = float('-inf')
    stack = deque()
    point = ()
    for i in range(N):
        for j in range(min(K, P)):
            point = (i, values[i][j], P - j - 1)
            stack.append(point)
    while stack:
        r, v, left = stack.popleft()
        if left == 0:
            maxvalues = max(maxvalues, v)
        else:
            if (N - r - 1) * K < left:
                continue
            if (N - r - 1) * K < left:
                maxvalues = (maxvalues, sum([values[i][-1] for i in range(r + 1, N)]))
            for i in range(r + 1, N):
                for j in range(min(K, left)):
                    stack.append((i, v + values[i][j], left - j - 1))
    return maxvalues




import numpy as np

T = int(input())
for t in range(T):
    N, K, P = map(int, input().split(' '))
    beauty = [[] for _ in range(N)]
    for i in range(N):
        beauty[i] = list(map(int, input().split(' ')))
    beauty = np.cumsum(beauty, axis=1)
    result = dpsolve(N, P, K, beauty)
    print('Case #%d: %d' % (t + 1, result))
