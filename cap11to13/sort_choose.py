from LinkedQueue import LinkedQueue
import math
import random


def merge(S1,S2,S):
    i=j=0
    while i+j<len(S):
        if j==len(S2) or (i<len(S1) and S1[i]<S2[j]):
            #j==len(S2)时，S2已经超出索引范围了
            S[i+j]=S1[i]
            i+=1
        else:
            S[i+j]=S2[j]
            j+=1


def merge_sort(S):
    n=len(S)
    if n<2:
        return
    mid=n//2
    S1=S[0:mid]
    S2=S[mid:]
    merge_sort(S1)
    merge_sort(S2)
    merge(S1,S2,S)


def merge1(S1,S2,S):
    while not S1.is_empty() and not S2.is_empty():
        if S1.first()<S2.first():
            S.enqueue(S1.dequeue())
        else:
            S.enqueue(S2.dequeue())
    while not S1.is_empty():
        S.enqueue(S1.dequeue())
    while not S2.is_empty():
        S.enqueue(S2.dequeue())


def merge_sort1(S):
    n=len(S)
    if n<2:
        return
    S1=LinkedQueue()
    S2=LinkedQueue()
    while len(S1)<n//2:
        S1.enqueue(S.dequeue())
    while not S.is_empty():
        S2.enqueue(S.dequeue())
    merge_sort1(S1)
    merge_sort1(S2)
    merge1(S1,S2,S)


def merge2(src,result,start,inc):
    end1=start+inc
    end2=min(start+2*inc,len(src))
    x,y,z=start,start+inc,start
    while x<end1 and y<end2:
        if src[x]<src[y]:
            result[z]=src[x];x+=1
        else:
            result[z]=src[y];y+=1
        z+=1
    if x<end1:
        result[z:end2]=src[x:end1]
    elif y<end2:
        result[z:end2]=src[y:end2]


def merge_sort2(S):
    n=len(S)
    logn=math.ceil(math.log(n,2))
    src,dest=S,[None]*n
    for i in (2**k for k in range(logn)):
        for j in range(0,n,2*i):
            merge2(src,dest,j,i)
        src,dest=dest,src
    if S is not src:
        S[0:n]=src[0:n]



def quick_sort(S):
    n=len(S)
    if n<2:
        return
    p=S.first()
    L=LinkedQueue()
    E=LinkedQueue()
    G=LinkedQueue()
    while not S.is_empty():
        if S.first()<p:
            L.enqueue(S.dequeue())
        elif p<S.first():
            G.enqueue(S.dequeue())
        else:
            E.enqueue(S.dequeue())
    quick_sort(L)
    quick_sort(G)
    while not L.is_empty():
        S.enqueue(L.dequeue())
    while not E.is_empty():
        S.enqueue(E.dequeue())
    while not G.is_empty():
        S.enqueue(G.dequeue())


def inplace_quick_sort(S,a,b):
    if a>b:return
    pivot=S[b]
    left=a
    right=b-1
    while left<=right:
        while left<=right and S[left]<pivot:
            left+=1
        while left<=right and pivot<S[right]:
            right-=1
        if left<=right:
            S[left],S[right]=S[right],S[left]
            left,right=left+1,right-1
    S[left],S[right]=S[b],S[left]
    inplace_quick_sort(S,a,left-1)
    inplace_quick_sort(S,left+1,b)



def quick_select(S,k):
    if len(S)==1:
        return
    pivot=random.choice(S)
    L=[x for x in S if S<pivot]
    E=[x for x in S if x==pivot]
    G=[x for x in S if x>pivot]
    if k<len(L):
        return quick_select(L,k)
    if k<=len(L)+len(E):
        return pivot
    else:
        j=k-len(L)-len(E)
        return quick_select(G,j)
    
