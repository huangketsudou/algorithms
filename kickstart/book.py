def book(N,M,Q,P,R):
    result=0
    for i in R:
        j=i
        while j<=N:
            if j not in P:
                result+=1
            j+=i
    return result


def book2(N,M,Q,P,R):
    result=0
    for i in R:
        tmp=N//i
        substract=sum([1 for j in P if j%i==0])
        result+=tmp-substract
    return result

# T=int(input())
# for t in range(T):
#     N,M,Q=map(int,input().split(' '))
#     P=list(map(int,input().split(' ')))
#     R=list(map(int,input().split(' ')))
#     result=book2(N,M,Q,P,R)
#     print('Case #%d: %d' %(t+1,result))


def equation(N,M,A):
    string=[]
    for i in A:
        string.append('{:064b}'.format(i))
    zeroindigit=[0]*64
    b=list(zip(*string))
    for i in range(len(b)):
        zeroindigit[i]=b[i].count('0')
    j=0
    k=0
    tmp=0
    while j<64:
        digitsum=zeroindigit[j]*2**(64-j-1)
        if tmp+digitsum<M:
            tmp+=digitsum
            k+=2**(64-j-1)
        else:
            tmp+=(N-zeroindigit[j])*2**(64-j-1)
        j+=1
    if k==0:
        if sum(A)<=M:
            return k
        else:
            return -1
    return result


T=int(input())
for t in range(T):
    N,M=map(int,input().split(' '))
    A=list(map(int,input().split(' ')))
    result=equation(N,M,A)
    print('Case #%d: %d' % (t + 1, result))
