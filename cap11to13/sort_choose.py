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
    
import sys
import numpy as np

# 2018年11月16日
# 黄杰栋


# 查找元素索引的二分法
def binary_search(list, item):
    low = 0
    high = len(list) - 1

    while low <= high:
        mid = (low + high) // 2
        guess = list[mid]
        if guess == item:
            return mid
        elif guess > item:
            high = mid - 1
        else:
            low = mid + 1
    return None


my_list = [1, 3, 5, 7, 9]
print(binary_search(my_list, 3))
print(binary_search(my_list, -1))


# ------------------------------------------------------
# 2018年11月17日


# 简单的排序法-可选由大到小排列或者由小到大排列
def findMinimum(arr):
    Minimum = arr[0]
    Minimum_index = 0
    for i in range(1, len(arr)):
        if arr[i] < Minimum:
            Minimum = arr[i]
            Minimum_index = i
    return Minimum_index


def findMaximum(arr):
    Maximum = arr[0]
    Maximum_index = 0
    for i in range(1, len(arr)):
        if arr[i] > Maximum:
            Maximum = arr[i]
            Maximum_index = i
    return Maximum_index


#
#
def selectionSort(arr, how):
    newArr = []
    f_s = 'find' + how.capitalize() + "imum"
    try:
        fun = getattr(sys.modules[__name__], f_s)  # 判断函数可用的办法
        if callable(fun):
            for i in range(len(arr)):
                target = fun(arr)
                newArr.append(arr.pop(target))
            return newArr
    except AttributeError:
        return "the function you call does not exist."


print(selectionSort([5, 3, 6, 2, 10], "mok"))


# -------------------------------------------------------
# 求解最大公约数


def CAL(a, b):
    rem = a % b
    if rem == 0:
        return b
    else:
        return CAL(b, rem)


def GCD(a, b):
    if a == b:
        print('the two numbers are the equal.')
        return a
    maxnum = max(a, b)
    minnum = min(a, b)
    if minnum == 0:
        return maxnum
    else:
        return CAL(maxnum, minnum)


print(GCD(8, 4))


# --------------------------------------------------------------
# 用递归实现sum函数


def sum_self(iter):
    if not iter:
        return 0
    out_number = iter.pop()
    return out_number + sum_self(iter)


print(sum_self([1, 2, 3, 4, 5]))


# ------------------------------------------------------------------
# 递归计数
def count_num(iter):
    if not iter:
        return 0
    iter.pop()
    return 1 + count_num(iter)


print(count_num([1, 2, 1, 2, 1, 9]))


# ------------------------------------------------------------------
# 取最大值
def get_max(iter):
    if len(iter) == 1 or len(iter) == 0:
        return 'There are no enough numbers to compare.'
    if len(iter) == 2:
        return iter[0] if iter[0] > iter[1] else iter[1]
    return iter[0] if iter[0] > get_max(iter[1:]) else get_max(iter[1:])


print(get_max([1, 5, 78, 9]))


# ------------------------------------------------------------------
# 快速排序


def quicksort(array):
    if len(array) < 2:
        return array
    else:
        pivot = array[0]
        less = [i for i in array[1:] if i <= pivot]
        greater = [i for i in array[1:] if i > pivot]
        return quicksort(less) + [pivot] + quicksort(greater)


print(quicksort([10, 5, 2, 3]))


def insertion_sort(array):
    for k in range(1,len(array)):
        cur=array[k]
        j=k
        while j>0 and array[j-1]>cur:
            array[j]=array[j-1]
            j-=1
        array[j]=cur
    return array

# ----------------------------------------------------------------------
# 散列表


# graph = {}
# graph['start'] = {}
# graph['start']['a'] = 6
# graph['start']['b'] = 2
# graph['a'] = {}
# graph['a']['fin'] = 1
# graph['b']={}
# graph['b']['a'] = 3
# graph['b']['fin'] = 5
# graph['fin'] = {}
# infinity=float('inf')
# costs={}
# costs['a']=6
# costs['b']=2
# costs['fin']=infinity
# parents={}
# parents['a']='start'
# parents['b']='start'
# parents['fin']=None
# processed=[]
#
#
# def find_lowest_cost_node(costs):
#     lowest_cost=float("inf")
#     lowest_cost_node=None
#     for node in costs:
#         cost=costs[node]
#         if cost<lowest_cost and node not in processed:
#             lowest_cost=cost
#             lowest_cost_node=node
#     return lowest_cost_node
#
# node=find_lowest_cost_node(costs)
# while node is not None:
#     cost=costs[node]
#     neighbors=graph[node]
#     for n in neighbors.keys():
#         new_cost=cost+neighbors[n]
#         if costs[n]>new_cost:
#             costs[n]=new_cost
#             parents[n]=node
#     processed.append(node)
#     node=find_lowest_cost_node(costs)
# print(parents)


#--------------------------------------------------------------------
#和上面的那个程序不能一起运行


graph={}
graph['start']={}
graph['start']['a']=5
graph['start']['b']=2
graph['a']={}
graph['a']['c']=4
graph['a']['d']=2
graph['b']={}
graph['b']['a']=8
graph['b']['d']=7
graph['c']={}
graph['c']['fin']=3
graph['c']['d']=6
graph['d']={}
graph['d']['fin']=1
graph['fin']={}


infinify=float('inf')
costs={}
costs['a']=5
costs['b']=2
costs['c']=infinify
costs['d']=infinify
costs['fin']=infinify


parents={}
parents['a']='start'
parents['b']='start'
parents['c']=None
parents['d']=None
parents['fin']=None


processed=[]


def find_lowest_cost_node(costs):
    lowest_cost=infinify
    lowest_cost_node=None
    for node in costs.keys():
        if node not in processed and costs[node]<lowest_cost:
            lowest_cost=costs[node]
            lowest_cost_node=node
    return lowest_cost_node


node=find_lowest_cost_node(costs)

while node is not None:
    cost=costs[node]
    neighbors=graph[node]
    for n in neighbors.keys():
        new_cost=cost+neighbors[n]
        if new_cost<costs[n]:
            costs[n]=new_cost
            parents[n]=node
    processed.append(node)
    node=find_lowest_cost_node(costs)

print(costs['fin'])


#---------------------------------------------------------------------
#2018年11月19日


states_needed={'mt','wa','or','id','nv','ut','ca','az'}
stations={}
stations['kone']={'id','nv','ut'}
stations['ktwo']={'wa','id','mt'}
stations['kthree']={'or','nv','ca'}
stations['kfour']={'nv','ut'}
stations['kfive']={'ca','az'}

final_stations=set()

while states_needed:
    best_station=None
    states_covered=set()
    for station,states in stations.items():
        covered=states_needed & states
        if len(covered)>len(states_covered):
            best_station=station
            states_covered=covered
    states_needed-=states_covered
    final_stations.add(best_station)

print(final_stations)


#--------------------------------------------------------------
#对比最长公共子串


def compared_word_longeststring(word_a,word_b):
    x=len(word_a)
    y=len(word_b)
    cell=np.zeros([x,y],dtype='int8')
    for i in range(x):
        for j in range(y):
            if word_a[i]==word_b[j]:
                cell[i][j]=cell[i-1][j-1]+1    #这里的cell[-1][-1]指向的是数组的
            else:                             #最后一个数，并不存在索引超出范围
                cell[i][j]=0                   #的问题。
    return np.max(cell)

a=compared_word_longeststring('abcd','abbd')
print(a)


#---------------------------------------------------------------
#对比最长的公共子序列


def compared_word_samealph(word_a,word_b):
    x=len(word_a)
    y=len(word_b)
    cell=np.zeros([x,y],dtype='int8')
    for i in range(x):
        for j in range(y):
            if word_a[i]==word_b[j]:
                cell[i][j]=cell[i-1][j-1]+1
            else:
                cell[i][j] = max(cell[i - 1][j], cell[i][j - 1])
    return np.max(cell)


b=compared_word_samealph('sdwe','dwae')
print(b)


