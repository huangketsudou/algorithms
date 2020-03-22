def solve(number, maxvalue, house):
    house.sort()
    costed = 0
    res = 0
    for i in range(number):
        costed += house[i]
        if costed<= maxvalue:
            res += 1
        else:
            return res
    return res

T = int(input())
for t in range(T):
    N, B = map(int, input().split(' '))
    A = list(map(int, input().split(' ')))
    result = solve(N, B, A)
    print('Case #%d: %d' % (t + 1, result))
