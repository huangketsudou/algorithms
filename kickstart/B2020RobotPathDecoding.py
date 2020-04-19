from collections import defaultdict, deque


def solve(path):
#只能解决小数据集
    border = 10 ** 9
    w = h = 0
    dictionary = defaultdict(int)
    d = {'N': 0, "S": 1, "E": 2, "W": 3}
    stack = deque()
    N = len(path)
    tmp = ''
    for i in range(N):
        if tmp=='':
            tmp+=path[i]
        elif path[i]=='(':
            stack.append(tmp)
            tmp=''
        elif tmp.isdigit() != path[i].isdigit():
            stack.append(tmp)
            tmp=path[i]
        elif path[i]==')':
            while stack and stack[-1].isdigit() == tmp.isdigit():
                tmp=stack.pop()+tmp
            if stack:
                tmp=int(stack.pop())*tmp
        else:
            tmp+=path[i]

    while stack:
        tmp=stack.pop()+tmp
    north=tmp.count('N')
    south=tmp.count('S')
    east=tmp.count('E')
    west=tmp.count('W')
    w=(east-west+1) % border
    h=(south-north+1) % border
    if w==0:w=border
    if h==0:h=border
    return w, h


T = int(input())
for t in range(T):
    path = input()
    w, h = solve(path)
    print('Case #%d: %d %d' % (t + 1, w, h))
#---------------------------------------------------------


def f(s):
    i = 0
    x, y = 0, 0
    while i < len(s):
        c = s[i]
        if c == 'N': y -= 1
        elif c == 'S': y += 1
        elif c == 'E': x += 1
        elif c == 'W': x -= 1
        else:
            r = int(c)
            j = i+2
            lvl = 1
            sub = ''
            while lvl:
                if s[j] == '(': lvl += 1
                if s[j] == ')': lvl -= 1
                sub += s[j]
                j += 1
            dx, dy = f(sub[:-1])
            x += dx * r
            y += dy * r
            i = j-1
        i += 1
    return x, y
    

M = int(1e9)
for t in range(input()):
    s = raw_input().strip()
    x, y = f(s)
    x %= M
    y %= M
    print('Case #%d: %d %d' % (t+1, x+1, y+1))
    
