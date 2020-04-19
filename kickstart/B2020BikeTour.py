def solve(N, H):
    ans = 0
    i = 1
    while i < N - 1:
        if H[i - 1] < H[i] and H[i] > H[i + 1]:
            ans+=1
        i+=1

    return ans


T = int(input())
for t in range(T):
    N = int(input())
    H = list(map(int, input().split(' ')))
    result=solve(N,H)
    print('Case #%d: %d' % (t + 1, result))
