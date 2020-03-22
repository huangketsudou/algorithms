from copy import deepcopy
import math
import numpy as np
from numba import jit,vectorize
from collections import defaultdict


# 使一个十进制数没有奇数位


def searchfirstEven(number: int) -> int:
    i = 1
    marked = 0
    while number:
        res = number % 10
        if res & 1:
            marked = i
        number = number // 10
        i += 1
    return marked


def getSmallnumber(number, firsteven):
    marked = firsteven
    divisor = number
    res = 0
    while marked > 0:
        res = divisor % 10
        divisor = divisor // 10
        marked -= 1
    numberofeight = 0
    resEven = firsteven
    while resEven - 2 >= 0:
        numberofeight += 8 * pow(10, resEven - 2)
        resEven -= 1
    return divisor * pow(10, firsteven) + (res - 1) * pow(10, firsteven - 1) + numberofeight


def getBignumber(number, firsteven):
    marked = firsteven
    divisor = number
    res = 0
    while marked > 0:
        res = divisor % 10
        divisor = divisor // 10
        marked -= 1
    if res != 9:
        return divisor * pow(10, firsteven) + (res + 1) * pow(10, firsteven - 1)
    else:
        carry = 0
        while divisor % 10 == 8:
            carry += 1
            divisor = divisor // 10
        return divisor * pow(10, firsteven + carry) + 2 * pow(10, carry + firsteven)


def Evendigit(number: int):
    if number < 0:
        return 0
    firstEven = searchfirstEven(number)
    if firstEven == 0:
        return 0
    smallnumber = getSmallnumber(number, firstEven)
    bignumber = getBignumber(number, firstEven)
    return min(bignumber - number, number - smallnumber)


# print(Evendigit(42))


def LuckyDip(data, k):
    if k < 0:
        return None
    if type(data) != list and len(data) == 0:
        return None
    data.sort()
    length = len(data)
    summary = sum(data)
    E = []
    E.append(summary / length)
    i = 1
    while i <= k:
        datacopy = deepcopy(data)
        j = 0
        while len(datacopy) > 0 and datacopy[0] < E[-1]:
            datacopy.pop(0)
            j += 1
        probility = (j * E[-1] + sum(datacopy)) / length
        E.append(probility)
        i += 1
    return E[-1]


# print(LuckyDip([16,11,7,4,1],3))


def gethash(start, end, alpha):
    hardseed = 123
    res = ord(start) * hardseed + ord(end)
    for i in range(26):
        res = res * hardseed + alpha[i]
    return res


def ScrambledWords():
    numberofwords = int(input())
    n = 0
    haxi = {}
    l = set()
    words = input().split(' ')
    while n < numberofwords:
        word = words[n]
        length = len(word)
        alpha = [0] * 26
        for i in range(length):
            alpha[ord(word[i]) - ord('a')] += 1
        haxi[gethash(word[0], word[-1], alpha)] = \
            haxi.setdefault(gethash(word[0], word[-1], alpha), 0) + 1
        l.add(length)
        n += 1
    S1, S2, N, A, B, C, D = input().split(' ')
    N = int(N)
    A = int(A)
    B = int(B)
    C = int(C)
    D = int(D)
    S = []
    numseed = []
    numseed.append(ord(S1))
    numseed.append(ord(S2))
    S.append(S1)
    S.append(S2)
    for i in range(2, N):
        numseed.append((A * numseed[i - 1] + B * numseed[i - 2] + C) % D)
        S.append(chr(numseed[i] % 26 + 97))
    S = ''.join(S)
    count = 0
    for i in l:
        if i > N:
            continue
        alpha = [0] * 26
        j = 0
        while j < i:
            alpha[ord(S[j]) - ord('a')] += 1
            j += 1
        subhash = gethash(S[0], S[i - 1], alpha)
        if subhash in haxi:
            count += haxi[subhash]
            del haxi[subhash]
        j = i
        while j < N:
            alpha[ord(S[j]) - ord('a')] += 1
            alpha[ord(S[j - i]) - ord('a')] -= 1
            subhash = gethash(S[j - i + 1], S[j], alpha)
            if subhash in haxi:
                count += haxi[subhash]
                del haxi[subhash]
            j += 1
    return count


def scramblemain():
    T = int(input())
    for t in range(T):
        result = ScrambledWords()
        print('Case #%d: %d' % (t + 1, result))


# scramblemain()


def NoNine(L, F):
    def solve(N):
        L = len(str(N))
        res = 0
        for i in range(L):
            if i < L - 1:
                res += int(str(N)[i]) * (9 ** (L - 2 - i)) * 8
            else:
                for i in range(N - N % 10, N + 1):
                    if i % 9 > 0:
                        res += 1
        return res

    print(solve(L))

    return solve(F) - solve(L) + 1


# print(NoNine(1234, 102))


def NoNine2(L, F):
    def find(x):
        res = 0
        for i in range(x - x % 10, x + 1):
            if i % 9 != 0:
                res += 1
        x /= 10
        mid = 0
        i = 0
        while x:
            mid += int(x % 10) * 9 ** i
            i += 1
            x /= 10
        mid = mid << 3
        return mid

    return find(F) - find(L) + 1


def planetdistance():
    T = int(input('input T cases ').split()[0])
    for n in range(T):
        N = int(input('input number of planets '))
        planet = defaultdict(list)
        for i in range(N):
            x, y = map(int, input('x and y').split(' '))
            planet[x].append(y)
            planet[y].append(x)
        queue = []
        degree = [0] * (N + 1)
        for i in range(1, N + 1):
            degree[i] = len(planet[i])
            if len(planet[i]) == 1:
                queue.append(i)
        distance = [0] * (N + 1)
        while len(queue) != 0:
            node = queue.pop(0)
            distance[node] = -1
            for i in range(len(planet[node])):
                v = planet[node][i]
                degree[v] -= 1
                if degree[v] == 1:
                    queue.append(v)
        for i in range(1, N + 1):
            if distance[i] == 0:
                queue.append(i)

        # 两个方法只能运行一个
        # BFS方法,queue初始为环，利用append当作队列
        while len(queue) != 0:
            node = queue.pop(0)
            for i in range(len(planet[node])):
                v = planet[node][i]
                if distance[v] == -1:
                    distance[v] = distance[node] + 1
                    queue.append(v)

        # DFS，queue初始为环，利用insert当作栈
        while len(queue) != 0:
            node = queue.pop(0)
            for i in range(len(planet[node])):
                v = planet[node][i]
                if distance[v] == -1:
                    distance[v] = distance[node] + 1
                    queue.insert(0, v)

        print(("Case #{}: " + ' '.join(map(str, distance[1:]))).format(n + 1))


# planetdistance()


def kickstartalarm():
    N, K, x1, y1, C, D, E1, E2, F = map(int, input('input N,K,x1,y1,C,D,E1,E2,F').split(' '))
    x = [x1]
    y = [y1]
    A = [x1 + y1]
    for i in range(1, N):
        x.append((C * x[i - 1] + D * y[i - 1] + E1) % F)
        y.append((D * x[i - 1] + C * y[i - 1] + E2) % F)
        A.append((x[i] + y[i]) % F)
    constnumber = 1000000007
    P = [0] * K
    res = 0
    for k in range(K):
        for l in range(N):
            for r in range(l, N):
                for j in range(l, r + 1):
                    P[k] += A[j] * pow(j - l + 1, k + 1)
    result = sum(P) % 1000000007
    return result, res


# print(kickstartalarm())


def kickstartalarmBig():
    constmod = 1000000007

    def power(base, n):
        if n == 0:
            return 1
        else:
            result = power(base, n >> 1)
            if n & 1:
                return ((result * result) % constmod * base) % constmod
        return (result * result) % constmod

    N, K, x1, y1, C, D, E1, E2, F = map(int, input().split(' '))
    x = [x1]
    y = [y1]
    A = [(x1 + y1) % F]
    for i in range(1, N):
        x.append((C * x[i - 1] + D * y[i - 1] + E1) % F)
        y.append((D * x[i - 1] + C * y[i - 1] + E2) % F)
        A.append((x[i] + y[i]) % F)

    result = 0
    lastsum = 0
    for i in range(1, N + 1):
        if i == 1:
            lastsum += K
        else:
            # 费马小定理
            # (b/a)%n=(b*a**n-2)%n
            lastsum += i * (power(i, K) - 1) % constmod * power(i - 1, constmod - 2) % constmod
        lastsum %= constmod
        temp = lastsum * (N - i + 1) % constmod
        result += temp * A[i - 1] % constmod
        result %= constmod

    return result


# T = int(input())
# for t in range(T):
#     result = kickstartalarmBig()
#     print(("Case #{}: " + str(result)).format(t + 1))


# -------------------------------------------------------------


def solve(S, N, D, O):
    smallest = -math.inf
    summary = np.cumsum(S).tolist()
    r = 1
    on = 0
    mt = []
    answer = smallest
    for i in range(1, N + 1):
        while r <= N and on + (S[r] & 1) <= O:
            mt.append(summary[r])
            on += (S[r] & 1)
            r += 1
        mt.sort()
        print(mt)
        if r <= i:
            r = i + 1
        else:
            # summary[j]-sum[i-1]<=D
            # 找的是j
            it = np.searchsorted(np.array(mt), D + summary[i - 1] + 1)
            if it != 0:
                it -= 1
                answer = max(answer, mt[it] - summary[i - 1])
                print(answer)
            mt.remove(summary[i])
            on -= (S[i] & 1)
    if answer == smallest:
        return 'IMPOSSIBLE'
    else:
        return answer


def candies():
    T = int(input())
    for t in range(T):
        N, O, D = map(int, input().split(' '))
        x = [0, 0]
        x[0], x[1], A, B, C, M, L = map(int, input().split(' '))
        i = 2
        while i < N:
            x.append((A * x[-1] + B * x[-2] + C) % M)
            i += 1
        S = []
        for i in x:
            S.append(i)
        S.insert(0, 0)
        result = solve(S, N, D, O)
        if result != None:
            print(("Case #{}: " + str(result)).format(t + 1))
        else:
            print(("Case #{}: " + 'IMPOSSIBLE').format(t + 1))


# candies()


def fun(x, y):
    return x + y, y - x


def paragliding(p, h, x, y, K, N):
    vec1 = list(map(fun, p, h))[1:]
    vec1.sort(reverse=True)
    vec2 = list(map(fun, x, y))[1:]
    vec2.sort(reverse=True)
    id = 0
    ans = 0
    maxY = -math.inf
    for i in range(K):
        while id < N and vec1[id][0] >= vec2[i][0]:
            maxY = max(maxY, vec1[id][1])
            id += 1
        if maxY >= vec2[i][1]:
            ans += 1
    return ans


def paraglidingmain():
    T = int(input())
    for t in range(T):
        N, K = map(int, input().split(' '))
        p = [0] * (N + 1)
        h = [0] * (N + 1)
        x = [0] * (K + 1)
        y = [0] * (K + 1)
        p[1], p[2], A1, B1, C1, M1 = map(int, input().split(' '))
        h[1], h[2], A2, B2, C2, M2 = map(int, input().split(' '))
        x[1], x[2], A3, B3, C3, M3 = map(int, input().split(' '))
        y[1], y[2], A4, B4, C4, M4 = map(int, input().split(' '))
        for i in range(3, N + 1):
            p[i] = (A1 * p[i - 1] + B1 * p[i - 2] + C1) % M1 + 1
            h[i] = (A2 * h[i - 1] + B2 * h[i - 2] + C2) % M2 + 1
        for k in range(3, K + 1):
            x[k] = (A3 * x[k - 1] + B3 * x[k - 2] + C3) % M3 + 1
            y[k] = (A4 * y[k - 1] + B4 * y[k - 2] + C4) % M4 + 1
        print(('Case #{}: ' + str(paragliding(p, h, x, y, K, N))).format(t + 1))


# paraglidingmain()


def yogurt(N, K, A):
    # O(Nlog(N))
    result = 0
    A.sort()
    days = 0
    consume = 0
    for i in range(N):
        if A[i] > days and consume < K:
            result += 1
            consume += 1
            if consume == K:
                consume = 0
                days += 1
    return result


def yogurt2(N, K, A):
    # O(N)
    days = [0] * N
    for a in A:
        days[min(a, N) - 1] += 1
    result = 0
    for i in range(N - 1, -1, -1):
        if i > 0:
            days[i - 1] += max(days[i] - K, 0)
        result += min(days[i], K)
    return result


def yogurtmain():
    T = int(input())
    for t in range(T):
        N, K = map(int, input().split(' '))
        A = list(map(int, input().split(' ')))
        result = yogurt2(N, K, A)
        print('Case #%s: %s' % (t + 1, result))


# yogurtmain()


def milktea(N, M, P, friends, forbidden):
    costofoption = [0] * P
    for i in range(P):
        costofoption[i] = sum(f[i] == '1' for f in friends)
    best = [[0, '']]
    for i in range(P):
        cur = []
        for cost, typ in best:
            cur.append([cost + costofoption[i], typ + '0'])
            cur.append([cost + N - costofoption[i], typ + '1'])
        best = sorted(cur)[:M + 1]
    for cost, typ in best:
        if typ not in forbidden:
            return cost


def milkteamain():
    T = int(input())
    for t in range(T):
        N, M, P = map(int, input().split(' '))
        friends = [input().strip() for i in range(N)]
        forbidden = set(input().strip() for i in range(M))
        result = milktea(N, M, P, friends, forbidden)
        print(('Case #%d: %d') % (t + 1, result))


# milkteamain()

def sign(string, start, end, alpha):
    current = ord(string[end - 1]) - ord('A')
    alpha[current] += 1
    result = ''
    for i in range(26):
        result += (str(alpha[i]) + ',')  # 防止有两位数的出现
    return result


def commonanagrams(L, A, B):
    Bset = set()
    for i in range(L):
        alpha = [0] * 26
        for j in range(i + 1, L + 1):
            Bset.add(sign(B, i, j, alpha))
    result = 0
    for i in range(L):
        alpha = [0] * 26
        for j in range(i + 1, L + 1):
            if sign(A, i, j, alpha) in Bset:
                result += 1
    return result


def commonAnagramsmain():
    T = int(input())
    for t in range(T):
        L = int(input())
        A = input().strip()
        B = input().strip()
        result = commonanagrams(L, A, B)
        print('Case #%d: %d' % (t + 1, result))


# commonAnagramsmain()

class Edge:
    def __init__(self):
        self.to = -1
        self.length = -1


def update(l, x, y, edges):
    if edges[x].to == -1:
        edges[x].to = y
        edges[x].length = l
    else:
        if edges[x].length > l:
            edges[x].to = y
            edges[x].length = l


def dfs(v, G, visited):
    if visited[v]:
        return
    visited[v] = 1
    for i in range(len(G[v])):
        dfs(G[v][i], G, visited)


def villagesmain():
    # 分析之后必定无环
    T = int(input())
    for t in range(T):
        V, E = map(int, input().split(' '))
        edges = [Edge() for i in range(V + 1)]
        zeros = set()
        for i in range(E):
            x, y, l = map(int, input().split(' '))
            update(l, x, y, edges)
            update(l, y, x, edges)
            if l == 0:
                zeros.add(x)
                zeros.add(y)
        G = [[] for i in range(V + 1)]
        for i in range(1, V + 1):
            G[i].append(edges[i].to)
            G[edges[i].to].append(i)
        count = 0
        visited = [0] * (V + 1)
        for i in range(1, V + 1):
            if not visited[i]:
                dfs(i, G, visited)
                # 可能是个非连通图
                count += 1
        # 除了最短边位0的点外，还有连接到这些点的点
        for i in range(1, V + 1):
            if edges[i].to in zeros:
                count += 1
        # -len(zeros)求出有几个点练到了有边为0的点上
        count -= len(zeros)
        print('Case #%d: %d' % (t + 1, 2 ** count))


# villagesmain()


def producttriple(N, A):
    # 统计计数问题
    number = defaultdict(int)
    for i in A:
        number[i] += 1
    result = 0
    for i in range(N):
        for j in range(i + 1, N):
            if A[i] == 0 or A[j] == 0 or A[i] == 1 or A[j] == 1:
                continue
            if A[i] * A[j] > 200000:  # 题目规定
                continue
            result += number[A[i] * A[j]]
    result += number[1] * (number[1] - 1) * (number[1] - 2) / 6
    for key, value in number.items():
        if key != 0 and key != 1:
            result += number[1] * value * (value - 1) / 2
    # for i in range(2,200001):
    #     result+=number[1]*number[i]*(number[i]-1)/2
    result += number[0] * (number[0] - 1) * (number[0] - 2) / 6
    result += number[0] * (number[0] - 1) * (N - number[0]) / 2
    return result


def productmain():
    T = int(input())
    for t in range(T):
        N = int(input())
        A = list(map(int, input().split(' ')))
        result = producttriple(N, A)
        print("Case #%d: %d" % (t + 1, result))


# productmain()


def combiningclasses(N, Q, L, R, K):
    a = [0] * (2 * N)
    total = 0
    for i in range(1, N + 1):
        a[total] = L[i]
        total += 1
        a[total] = R[i]
        total += 1
    a.sort()
    b = list(set(a))
    total = len(b)
    for i in range(1, N + 1):
        L[i] = np.searchsorted(np.array(b), L[i])
        R[i] = np.searchsorted(np.array(b), R[i])
    f = [0] * (total)
    for i in range(1, N + 1):
        f[L[i]] += 1
        f[R[i]] -= 1
    for i in range(1, total):
        f[i] += f[i - 1]
    g = [0] * total
    for i in range(total - 1, -1, -1):
        if i == total - 1:
            g[i] = f[i] + g[i]
            continue
        g[i] = g[i + 1] + f[i] * (b[i + 1] - b[i])
    result = 0
    for i in range(1, Q + 1):
        if K[i] > g[0]:
            continue
        ret = 0
        l = 0
        r = total - 1
        while l <= r:
            mid = (l + r) // 2
            if g[mid] >= K[i]:
                ret = mid
                l = mid + 1
            else:
                r = mid - 1
        tmp = b[ret] + (g[ret] - K[i]) / f[ret]
        result += tmp * i
    return result


def combinemain():
    T = int(input())
    for t in range(T):
        N, Q = map(int, input().split(' '))
        X = [0]
        Y = [0]
        Z = [0]
        X1, X2, A1, B1, C1, M1 = map(int, input().split(' '))
        Y1, Y2, A2, B2, C2, M2 = map(int, input().split(' '))
        Z1, Z2, A3, B3, C3, M3 = map(int, input().split(' '))
        X.extend([X1, X2])
        Y.extend([Y1, Y2])
        Z.extend([Z1, Z2])
        for i in range(3, N + 1):
            X.append((A1 * X[i - 1] + B1 * X[i - 2] + C1) % M1)
            Y.append((A2 * Y[i - 1] + B2 * Y[i - 2] + C2) % M2)
        for i in range(3, Q + 1):
            Z.append((A3 * Z[i - 1] + B3 * Z[i - 2] + C3) % M3)
        L = [min(X[i], Y[i]) + 1 for i in range(N + 1)]
        R = [max(X[i], Y[i]) + 1 for i in range(N + 1)]
        K = [i + 1 for i in Z]
        print(K)
        result = combiningclasses(N, Q, L, R, K)
        print('Case #%d: %d' % (t + 1, result))


# combinemain()


def button(N, P, p):
    seen = set()
    all = 2 ** N
    for s in sorted(p, key=len):
        if any(s[:i + 1] in seen for i in range(len(s))):
            continue
        seen.add(s)
        all -= 2 ** (N - len(s))
    return all


def buttonmain():
    T = int(input())
    for t in range(T):
        N, P = map(int, input().split(' '))
        p = []
        for i in range(P):
            p.append(input().strip())
        result = button(N, P, p)
        print("Case #%d: %d" % (t + 1, result))


# buttonmain()


def mural():
    T = int(input())
    for t in range(T):
        N = int(input())
        a = list(map(int, input().strip()))
        half = (N + 1) // 2
        result = answer = sum(a[0: half])
        for i in range(N - half - 1):
            answer += -a[i] + a[i + half]
            result = max(answer, result)
        print('Case #%d: %d' % (t + 1, result))


# mural()
from functools import reduce
from decimal import Decimal

def power(base, n, constmod):
    if n == 0:
        return 1
    else:
        result = power(base, n >> 1, constmod)
        if n & 1:
            return ((result * result) % constmod * base) % constmod
    return (result * result) % constmod


def letmecountmain(N, M):
    constmod = 1000000007
    com = [0] * (M + 1)
    double = [0] * (M + 1)
    sit = [0] * (M + 1)
    i = 0
    while i <= M:
        if i == 0:
            double[i] = 1
            # com[i] = 1 % constmod
        else:
            # double[i] = power(constmod - 2, i, constmod)
            double[i] = (double[i-1]*-2)% constmod
            # com[i] = ((M - i + 1) * power(i, constmod - 2, constmod) * com[i - 1]) % constmod
        i += 1
    #C(m,j)是对称的
    i=0
    while i<=(M+1)//2:
        if i==0:
            com[i]=com[M-i]=1%constmod
        else:
            com[i]=com[M-i]=((M - i + 1) * power(i, constmod - 2, constmod) * com[i - 1]) % constmod
        i+=1
    i = M
    while i >= 0:
        if i == M:
            sit[i] = reduce(lambda x, y: x * y, range(1, 2 * N - M + 1)) % constmod
        else:
            sit[i] = (sit[i + 1] * (2 * N - i)) % constmod
        i -= 1
    sum = 0
    for x, y, z in zip(double, sit, com):
        sum += (x * y * z) % constmod
    sum %= constmod
    return sum


def letmecount():
    T = int(input())
    for t in range(T):
        N, M = map(int, input().split(' '))
        result = letmecountmain(N, M)
        print("Case #%d: %d" % (t + 1, result))


# letmecount()


def circuit(R,C,K,V):
    diff=np.zeros([R,C,C],dtype=int)
    for i in range(R):
        for j in range(C):
            minimum=V[i][j]
            maximum=V[i][j]
            for k in range(j,C):
                minimum=min(minimum,V[i][k])
                maximum=max(maximum,V[i][k])
                diff[i][j][k]=maximum-minimum
    answer=0
    for i in range(C):
        for j in range(i,C):
            now=0
            for k in range(R):
                if diff[k][i][j]<=K:
                    answer=max(answer,(j-i+1)*(k-now+1))
                else:
                    now=k+1
    return answer



def circuitmain():
    T=int(input())
    for t in range(T):
        R,C,K=map(int,input().split(' '))
        V=[list(map(int,input().split(' '))) for i in range(R)]
        result=circuit(R,C,K,V)
        print("Case #%d: %d" % (t+1,result))


# circuitmain()


def catch(N,K,P,A):
    INf=1<<31
    dp=np.zeros([1001,1001,2])
    vector=[[] for i in range(1010)]
    for i in range(N):
        vector[A[i]].append(P[i])
    for i in range(1,1001):
        vector[i].sort()
    for i in range(1001):
        for j in range(N+1):
            dp[i][j][0]=dp[i][j][1]=INf
    dp[0][0][1]=dp[0][0][1]=0
    for i in range(1000):
        for j in range(N+1):
            if dp[i][j][0]==INf and dp[i][j][1]==INf:
                continue
            sz=len(vector[i+1])
            for x in range(sz+1):
                tmp=0 if x==0 else vector[i+1][x-1]
                dp[i + 1][j + x][0] = min(dp[i + 1][j + x][0], dp[i][j][0] + 2 * tmp)
                dp[i + 1][j + x][1] = min(dp[i + 1][j + x][1], dp[i][j][0] + tmp)
                dp[i + 1][j + x][1] = min(dp[i + 1][j + x][1], dp[i][j][1] + 2 * tmp)
    return dp[1000][K-1][1]


def catchmain():
    T=int(input())
    for t in range(T):
        N,K=map(int,input().split(' '))
        P=list(map(int,input().split(' ')))
        A=list(map(int,input().split(' ')))
        print(A)
        print(P)
        result=catch(N,K,P,A)
        print("Case #%d: %d" %(t+1,result))

catchmain()
