def solve(N, D, X):
    dp = [0] * N
    for i in range(N - 1, -1, -1):
        dp[i] = X[i] * (D // X[i])
        D=dp[i]
    return dp[0]

#溢出问题

T = int(input())
for t in range(T):
    N, D = map(int, input().split(' '))
    X = list(map(int, input().split(' ')))
    result = solve(N, D, X)
    print('Case #%d: %d' % (t + 1, result))
