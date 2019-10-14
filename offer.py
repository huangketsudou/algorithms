from Tree import LinkedBinaryTree
import numpy as np
import random
from math import ceil
from functools import cmp_to_key


class Empty(Exception):
    pass


class LStack:
    class _Node:
        __slots__ = '_e', '_next'

        def __init__(self, e, next):
            self._e = e
            self._next = next

    def __init__(self):
        self._head = None
        self._size = 0

    def is_empty(self):
        return self._size == 0

    def __len__(self):
        return self._size

    def push(self, e):
        self._head = self._Node(e, self._head)
        self._size += 1

    def pop(self):
        if self.is_empty():
            raise Empty()
        answer = self._head._e
        self._head = self._head._next
        self._size -= 1
        return answer

    def top(self):
        if self.is_empty():
            raise Empty()
        return self._head._e


class LQueue:
    class _Node:
        __slots__ = '_e', '_next'

        def __init__(self, e, next):
            self._e = e
            self._next = next

    def __init__(self):
        self._head = None
        self._size = 0
        self._tail = None

    def is_empty(self):
        return self._size == 0

    def __len__(self):
        return self._size

    def first(self):
        if self.is_empty():
            raise Empty()
        return self._head._e

    def enqueue(self, e):
        newest = self._Node(e, None)
        if self.is_empty():
            self._head = newest
        else:
            self._tail._next = newest
        self._tail = newest
        self._size += 1

    def dequeue(self):
        if self.is_empty():
            raise Empty()
        answer = self._head._e
        self._head = self._head._next
        if self.is_empty():
            self._tail = None
        self._size -= 1
        return answer


class CQueue:
    class _Node:

        __slots__ = '_e', '_next'

        def __init__(self, e, next):
            self._e = e
            self._next = next

    def __init__(self):
        self._tail = None
        self._size = 0

    def __len__(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def first(self):
        if self.is_empty():
            raise Empty()
        return self._tail._next._e

    def enqueue(self, e):
        newest = self._Node(e, None)
        if self.is_empty():
            newest._next = newest
        else:
            newest._next = self._tail._next
            self._tail._next = newest
        self._tail = newest
        self._size += 1

    def dequeue(self):
        if self.is_empty():
            raise Empty()
        head = self._tail._next
        if self._size == 1:
            self._tail = None
        else:
            self._tail._next = head._next
        self._size -= 1
        return head._e

    def rotate(self):
        if self.is_empty():
            raise Empty()
        self._tail = self._tail._next


class TreeNode:
    def __init__(self, e):
        self.e = e
        self.left = None
        self.right = None


def construct_tree(preorder, inorder):
    if len(preorder) == 0:
        return None
    r = preorder[0]
    root = TreeNode(r)
    where_root = inorder.index(r)
    root.left = construct_tree(preorder[1:1 + where_root], inorder[:where_root])
    root.right = construct_tree(preorder[1 + where_root:], inorder[where_root + 1:])
    return root


def backorder(preorder, inorder):
    if len(preorder) != 0:
        root = preorder[0]
        root_index = inorder.index(root)
        for node in backorder(preorder[1:1 + root_index], inorder[:root_index]):
            yield node
        for node in backorder(preorder[1 + root_index:], inorder[root_index + 1:]):
            yield node
        yield root


# ------------------------------------------------------------------------
# 通过中序前序实现后序遍历
# for i in backorder([1,2,4,7,3,5,6,8],[4,7,2,1,5,3,8,6]):
#     print(i)


class DoubleLinkedQueue:

    def __init__(self):
        self._stack1 = LStack()  # 输入
        self._stack2 = LStack()  # 输出

    def push(self, e):
        while not self._stack2.is_empty():
            self._stack1.push(self._stack2.pop())
        self._stack1.push(e)

    def pop(self):
        while not self._stack1.is_empty():
            self._stack2.push(self._stack1.pop())
        if self._stack2.is_empty():
            raise Empty()
        return self._stack2.pop()


def fib(n):
    f = [0, 1, 1]
    if n <= 2:
        return f[n]
    for i in range(3, n + 1):
        f.append(f[i - 1] + f[i - 2])
    return f[n]


# print(fib(3))


def fib2(n):
    if n < 2:
        return (0, n)
    else:
        a, b = fib2(n - 1)
        return (a, a + b)


def BinarySearch(arr):
    if len(arr) < 2:
        return arr
    else:
        index = random.randint(0, len(arr))
        pivot = arr[index]
        less = [i for i in arr if i < pivot]
        greater = [i for i in arr if i > pivot]
        equaler = [i for i in arr if i == pivot]
    return BinarySearch(less) + equaler + BinarySearch(greater)


def get_smallest(reversedArr):
    if not reversedArr:
        return
    smaller = reversedArr.pop()
    while reversedArr:
        nextOne = reversedArr.pop()
        if nextOne > smaller:
            return smaller
        else:
            smaller = nextOne
    return smaller


# print(get_smallest([]))


def getSmallest(reversedArr):
    if not reversedArr:
        return
    first = 0
    last = len(reversedArr) - 1
    while last - first > 1:
        if reversedArr[first] < reversedArr[last]:
            return reversedArr[first]
        else:
            middle = (last + first) // 2
            if reversedArr[first] == reversedArr[last] == reversedArr[middle]:
                for i in range(first + 1, middle):
                    if reversedArr[i] < reversedArr[first]:
                        return reversedArr[i]
                first = middle
            elif reversedArr[first] <= reversedArr[middle]:
                first = middle
            elif reversedArr[middle] <= reversedArr[last]:
                last = middle
    return reversedArr[last]


# print(getSmallest([]))


def hasPath(arr, str, rows, cols):
    if not arr or not str or rows < 1 or cols < 1:
        return False
    hadfind = 0
    walk = [False] * rows * cols
    for i in range(rows):
        for j in range(cols):
            if find_path(arr, str, rows, cols, i, j, hadfind, walk):
                return True
    return False


def find_path(arr, string, rows, cols, i, j, hadfind, walk):
    if hadfind >= len(string):
        return True
    index = i * cols + j
    if i < 0 or j < 0 or i >= rows or j >= cols or arr[index] != string[hadfind] or walk[index]:
        return False
    walk[index] = True
    if find_path(arr, string, rows, cols, i + 1, j, hadfind + 1, walk) or find_path(arr, string, rows, cols, i - 1, j,
                                                                                    hadfind + 1, walk) or \
            find_path(arr, string, rows, cols, i, j + 1, hadfind + 1, walk) or find_path(arr, string, rows, cols, i,
                                                                                         j - 1,
                                                                                         hadfind + 1, walk):
        return True
    walk[index] = False
    return False


# k=['a','b','t','g','c','f','c','s','j','d','e','h']
# s='abtgs'
# r=1
# c=12
# print(hasPath(k,s,r,c))


def get_sum(i):
    over = 0
    while i > 0:
        over += i % 10
        i = i // 10
    return over


def walkAccessible(row, col, i, j, k, walk):
    over = get_sum(i) + get_sum(j)
    if 0 <= i < row and 0 <= j < col and over <= k and not walk[i][j]:
        return True
    return False


def RobotWalk(row, col, i, j, k, walked):
    count = 0
    if walkAccessible(row, col, i, j, k, walked):
        walked[i][j] = True
        count = 1 + RobotWalk(row, col, i - 1, j, k, walked) + \
                RobotWalk(row, col, i, j - 1, k, walked) + \
                RobotWalk(row, col, i, j + 1, k, walked) + \
                RobotWalk(row, col, i + 1, j, k, walked)
    return count


def Robot(row, col, k):
    if k < 0:
        return 0
    if row < 1 or col < 1:
        raise ValueError('Invalid!')
    walked = [[False] * col for _ in range(row)]
    count = RobotWalk(row, col, 0, 0, k, walked)
    print(walked)
    return count


# print(Robot(10,10,8))


def cutRole(length):
    if type(length) != int:
        raise TypeError('not an int')
    if length <= 1:
        return 0
    if length == 2:
        return 1
    if length == 3:
        return 2
    products = [0] * (length + 1)
    products[0] = 0
    products[1] = 1
    products[2] = 2
    products[3] = 3

    for i in range(4, length + 1):
        max = 0
        for j in range(1, i):
            if products[j] * products[i - j] > max:
                max = products[j] * products[i - j]
        products[i] = max

    return products[-1]


# print(cutRole(4))


def CutRole(length):
    if type(length) != int:
        raise TypeError('not an int')
    if length <= 1:
        return 0
    if length == 2:
        return 1
    if length == 3:
        return 2
    time = length // 3
    if length - time * 3 == 1:
        time -= 1
    return pow(3, time) * (length - 3 * time)


# print(CutRole(8))


def NumberOfOne(n):
    # 二进制下1的位数
    count = 0
    while n & 0xffffffff != 0:
        # 利用0xffffffff限值在32位之内
        count += 1
        n = n & (n - 1)
    return count


# print(NumberOfOne(-3))


# print(NumberOfOne(0xffff))


def ChangeN1ToN2(i, j):
    # 二进制下修改一个数为另一个数需要修改的位数
    n = i ^ j
    count = NumberOfOne(n)
    return count


# print(ChangeN1ToN2(10,13))


def PowerFunction(base, exponent):
    if exponent == 0:
        return 1
    if exponent == 1:
        return base
    result = PowerFunction(base, exponent >> 1)
    result *= result
    if (exponent & 1):
        result *= base
    return result


def powerBase(base, exponent):
    if exponent == 0 and base == 0:
        raise ValueError('both base and exponent are zero')
    if type(exponent) != int:
        raise ValueError('not an int')
    if exponent < 0:
        if base == 0:
            raise ValueError('exponent is minus and base is zero')
        absexponent = -exponent
        result = 1 / PowerFunction(base, absexponent)
    else:
        result = PowerFunction(base, exponent)
    return result


# print(powerBase(0,-6))


def power(base, exponent):
    if exponent == 0:
        return 1
    result = power(base, exponent >> 1)
    result *= result
    if exponent & 1:
        result *= base
    return result


def Print1ToNdigit(n):
    if n <= 0:
        return
    number = ['0'] * n
    while not Increment(number):
        PrintNumber(number)


def Increment(number):
    overFlow = False
    TakeOver = 0
    length = len(number)
    for i in range(length - 1, -1, -1):
        Sum = int(number[i]) + TakeOver
        if i == length - 1:  # 防止Print1ToNmaxdigit 输出空白的000
            Sum += 1
        if Sum >= 10:
            if i == 0:
                overFlow = True
            else:
                TakeOver = 1  # 仅有9999这种情况所有都要进位
                number[i] = '0'
        else:
            number[i] = str(Sum)
    return overFlow


def PrintNumber(number):
    length = len(number)
    skip = True
    for i in range(length):
        if skip and number[i] == '0':
            print('', end='')
            if i == length - 1:
                return
        else:
            skip = False
            print(number[i], end='')
    print('\n', end='')


# Print1ToNdigit(2)


def Print1ToNMaxdigit(n):
    if n <= 0:
        return
    number = ['0'] * n
    for i in range(0, 10):
        number[0] = str(i)
        Print1ToNdigitRecursively(number, n, 0)


def Print1ToNdigitRecursively(number, length, index):
    if index == length - 1:
        PrintNumber(number)
    else:
        for i in range(0, 10):
            number[index + 1] = str(i)
            Print1ToNdigitRecursively(number, length, index + 1)


# Print1ToNMaxdigit(2)


def bitPlus(a, b):
    while a:
        return bitPlus((a & b) << 1, a ^ b)
    return b


# print(bitPlus(-1,1))

class Node:
    def __init__(self, e, next=None):
        self._e = e
        self._next = next


n1 = Node(1)
n2 = Node(2, n1)
n3 = Node(3, n2)


def DeleteNode(head, i):
    if type(i) != Node or type(head) != Node:
        raise TypeError('not a Node')
    if i._next == None:
        if head == i:
            head = None
            return
        else:
            pivot = head
            while pivot._next != i:
                pivot = pivot._next
            pivot._next = None
            del i
    else:
        i._e = i._next._e
        next_point = i._next
        i._next = next_point._next
        del next_point


def match_pattern(string, pattern):
    if string == '' and pattern == '':
        return True
    if string == '' or pattern == '':
        return False
    return matchCore(string, pattern)


def matchCore(string, pattern):
    if string == pattern:
        return True
    if not pattern and string:
        return False
    if len(pattern) > 1 and pattern[1] != '*':
        if string and string[0] == pattern[0] or pattern[0] == '.':
            return matchCore(string[1:], pattern[1:])
    else:
        if string[0] == pattern[0] or pattern[0] == '.':
            return matchCore(string[1:], pattern) or matchCore(string[1:], pattern[2:]) or matchCore(string,
                                                                                                     pattern[2:])
        else:
            return matchCore(string, pattern[2:])
    return False


# print(matchCore('cb','a*cb'))


def isNumberic(string):
    if not isinstance(string, str):
        return False
    allowDot = True
    allowE = True
    for i in range(len(string)):
        if i == 0 and string[i] in 'Ee':
            return False
        if string[i] in '+-' and (i == 0 or string[i - 1] in 'Ee') and i < len(string) - 1:
            continue
        elif allowDot and string[i] == '.':
            allowDot = False
        elif allowE and string[i] in 'Ee':
            allowE = False
            allowDot = False
            if i == len(string) - 1:
                return False
        elif string[i] not in '0123456789':
            return False
    return True


# print(isNumberic([1,2]))


def arrangeEvenOdd(arr):
    if not arr:
        return
    pivot_odd = 0
    pivot_even = len(arr) - 1
    while pivot_even > pivot_odd:
        while (pivot_even > pivot_odd) and arr[pivot_odd] & 1:
            pivot_odd += 1
        while (pivot_even > pivot_odd) and not arr[pivot_even] & 1:
            pivot_even -= 1
        if pivot_odd < pivot_even:
            arr[pivot_odd], arr[pivot_even] = arr[pivot_even], arr[pivot_odd]
    return arr


# a = [2, 1, 4, 3]
# arrangeEvenOdd(a)
# print(a)


def FindKthToTail(k, Lhead):
    if k <= 0:
        return False
    if type(Lhead) != Node:
        return False
    pivot = Lhead
    target = Lhead
    for i in range(0, k - 1):
        if pivot._next != None:
            pivot = pivot._next
        else:
            return False
    while pivot._next != None:
        pivot = pivot._next
        target = target._next
    return target._e


def meetNode(Lhead):
    if type(Lhead) != Node:
        return None
    if Lhead._next == None:
        return None
    Nslow = Lhead._next
    Nfast = Lhead._next._next
    while Nfast != Nslow:
        if Nfast == None or Nslow == None or Nfast._next == None or Nfast._next == None:
            return None
        Nslow = Nslow._next
        Nfast = Nfast._next
    return Nfast


def detectN(Lhead):
    meetingNode = meetNode(Lhead)
    if not meetingNode:
        return 0
    pivot = meetingNode._next
    count = 1
    while pivot != meetingNode:
        pivot = pivot._next
        count += 1
    return count


def GateOfRing(Lhead):
    NodeOfRing = detectN(Lhead)
    if NodeOfRing == 0:
        return None
    Lfirst = Lhead
    Llast = Lhead
    while NodeOfRing != 0:
        Lfirst = Lfirst._next
        NodeOfRing -= 1
    while Lfirst != Llast:
        Lfirst = Lfirst._next
        Llast = Llast._next
    return Lfirst


def ReversedNode(Lhead):
    if type(Lhead) != Node:
        return None
    if Lhead._next == None:
        return Lhead
    reversedHead = None
    N = Lhead
    Npre = None
    while N != None:
        Nnext = N._next
        if N._next == None:
            reversedHead = N
        N._next = Npre
        Npre = N
        N = Nnext
    return reversedHead


def ConcatNode(N1head, N2head):
    if type(N1head) != Node or type(N2head) != Node:
        return None
    if N1head._e < N2head._e:
        mergeHead = N1head
        pivot1 = mergeHead._next
        pivot2 = N2head
    else:
        mergeHead = N2head
        pivot1 = mergeHead._next
        pivot2 = N1head
    RecordHead = mergeHead
    while pivot1 != None and pivot2 != None:
        if pivot1._e > pivot2._e:
            mergeHead._next = pivot1
            pivot1 = pivot1._next
            mergeHead = mergeHead._next
        else:
            mergeHead._next = pivot2
            pivot2 = pivot2._next
            mergeHead = mergeHead._next
    if pivot1 == None:
        mergeHead._next = pivot2
    if pivot2 == None:
        mergeHead._next = pivot1
    return RecordHead


def concatNodeRecursion(N1head, N2head):
    if N1head == None:
        return N2head
    elif N2head == None:
        return N1head
    if N1head._e < N2head._e:
        mergeHead = N1head
        mergeHead._next = concatNodeRecursion(N1head._next, N2head)
    else:
        mergeHead = N2head
        mergeHead._next = concatNodeRecursion(N1head, N2head._next)
    return mergeHead


def issubtree(T1Root, T2Root):
    '''
    子树搜索
    :param T1Root:
    :param T2Root:
    :return:
    '''
    found = False
    if T1Root and T2Root:
        found = RootEqual(T1Root, T2Root)
    if not found:
        found = issubtree(T1Root.left, T2Root)
    if not found:
        found = issubtree(T1Root.right, T2Root)
    return found


def RootEqual(T1Root, T2Root):
    if T2Root == None:
        return True
    if T1Root == None:
        return False
    if T1Root.e != T2Root.e:
        return False
    return RootEqual(T1Root.left, T2Root.left) and RootEqual(T1Root.right, T2Root.right)


def mirrorTree(TRoot):
    if TRoot == None:
        return
    if TRoot.left == None and TRoot.right == None:
        return
    TRoot.left, TRoot.right = TRoot.right, TRoot.left
    mirrorTree(TRoot.left)
    mirrorTree(TRoot.right)


# T3=Tree(3)
# T2=Tree(2)
# T1=Tree(1,T2)
#
# mirrorTree(T1)


def sysmetricTree(T1root, T2root):
    if T1root == None and T2root == None:
        return True
    if T1root == None or T2root == None:
        return False
    if T1root.e != T2root.e:
        return False
    return sysmetricTree(T1root.left, T2root.right) and sysmetricTree(T1root.right, T2root.left)


def ClockprintMatrix(arr, rows, cols):
    if not arr or rows == 0 or cols == 0:
        return
    start = 0
    while cols > start * 2 and rows > start * 2:
        ClockPrintRecur(arr, rows, cols, start)
        start += 1


def ClockPrintRecur(arr, rows, cols, start):
    endX = cols - 1 - start
    endY = rows - 1 - start

    for i in range(start, endX + 1):
        print(arr[start][i])

    if start < endY:
        for i in range(start + 1, endY):
            print(arr[i][endX])

    if start < endX and start < endY:
        for i in range(endX, start - 1, -1):
            print(arr[endY][i])

    if start < endX and start < endY - 1:
        for i in range(endY - 1, start, -1):
            print(arr[i][start])


# a=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
# ClockprintMatrix(a,4,4)


class MinStack:

    def __init__(self):
        self.dataStack = LStack()
        self.minstack = LStack()

    def is_empty(self):
        if self.dataStack.is_empty():
            return True

    def push(self, e):
        if self.is_empty():
            self.dataStack.push(e)
            self.minstack.push(e)
        else:
            self.dataStack.push(e)
            if e < self.min():
                self.minstack.push(e)
            else:
                self.minstack.push(self.min())

    def min(self):
        if not self.is_empty():
            return self.minstack.top()

    def pop(self):
        if self.is_empty():
            raise Empty()
        self.dataStack.pop()
        self.minstack.pop()

    def top(self):
        if not self.is_empty():
            return self.dataStack.top()


def isPopOrder(Push, Pop):
    flag = False
    if Push and Pop:
        start = 0
        out = 0
        assistStack = LStack()
        while out < len(Pop):
            while assistStack.is_empty() or assistStack.top() != Pop[out]:
                if start == len(Push):
                    break
                assistStack.push(Push[start])
                start += 1

            if assistStack.top() != Pop[out]:
                break

            assistStack.pop()
            out += 1
        if assistStack.is_empty() and out == len(Pop):
            flag = True
    return flag


# a=[1,2,3,4,5]
# b=[4,3,5,1,2]
# print(isPopOrder(a,b))

def PrintTopToBot(TreeHead):
    # 课本的基于数组表示的二叉树
    # 利用栈可以从后打到前
    if type(TreeHead) != TreeNode:
        raise Empty
    PrintQueue = LQueue()
    PrintQueue.enqueue(TreeHead)
    while not PrintQueue.is_empty():
        head = PrintQueue.dequeue()
        print(head.e)
        if head.left:
            PrintQueue.enqueue(head.left)
        if head.right:
            PrintQueue.enqueue(head.right)


# 基于数组表示的二叉树实现
def MatrixTree(THead, height):
    if type(THead) != TreeNode:
        return
    arr = [0] * (pow(2, height) - 1)
    MatrixTreeRecur(THead, arr, 0)
    return arr


def MatrixTreeRecur(Thead, arr, index):
    if Thead == None:
        return
    arr[index] = Thead.e
    left = 2 * index + 1
    right = 2 * index + 2
    MatrixTreeRecur(Thead.left, arr, left)
    MatrixTreeRecur(Thead.right, arr, right)


# T1=TreeNode(0)
# T2=TreeNode(1)
# T3=TreeNode(2)
# T4=TreeNode(3)
# T1.left=T2
# T1.right=T3
# T3.left=T4
#
# k=MatrixTree(T1,3)
# print(k)


def PrintTopToBotVer2(Thead):
    if type(Thead) != TreeNode:
        return
    if not Thead:
        return
    ToBePrint = 1
    nextlevel = 0
    PrintQueue = LQueue()
    PrintQueue.enqueue(Thead)
    while not PrintQueue.is_empty():
        PrintNode = PrintQueue.dequeue()
        print(PrintNode.e)
        if not PrintNode.left:
            PrintQueue.enqueue(PrintNode.left)
            nextlevel += 1
        if not PrintNode.right:
            PrintQueue.enqueue(PrintNode.right)
            nextlevel += 1
        ToBePrint -= 1
        if ToBePrint == 0:
            print('\n')
            ToBePrint = nextlevel
            nextlevel = 0


def ZPrintfromTopToBot(Thead):
    if type(Thead) != TreeNode:
        return
    PrintStack1 = LStack()
    PrintStack1.push(Thead)
    PrintStack2 = LStack()
    s = [PrintStack1, PrintStack2]
    current = 0
    next = 1
    while not s[0].is_empty() or not s[1].is_empty():
        node = s[current].pop()
        print(node.e)
        if current == 0:
            if node.left:
                s[next].push(node.left)
            if node.right:
                s[next].push(node.right)
        else:
            if node.right:
                s[next].push(node.right)
            if node.left:
                s[next].push(node.left)

        if s[current].is_empty():
            print('\n')
            current = 1 - current
            next = 1 - next


def KMP_fail(string):
    # 好理解，但是算法复杂度为O(m**2)
    n = len(string)
    fail = [0] * n
    i = 2
    while i <= n:
        part = string[:i]
        k = 1
        while k < i:
            if part[:k] == part[-k:]:
                fail[i - 1] = k
            k += 1
        i += 1
    return fail


# print(KMP_fail('amalgamation'))


def VerifySequenceOfBST(arr):
    if not arr:
        return False
    length = len(arr)
    root = arr[length - 1]
    i = 0
    while i < length - 1:
        if arr[i] > root:
            break
        i += 1
    j = i
    while j < length - 1:
        if arr[j] < root:
            return False
        j += 1
    left = True
    right = True
    if i > 0:
        left = VerifySequenceOfBST(arr[:i])
    if i < length - 1:
        print(arr[i:length - 1])
        right = VerifySequenceOfBST(arr[i:length - 1])
    return left and right


# print(VerifySequenceOfBST([5,7,6,8]))


def FindPath(TNode, expectedSum):
    if type(TNode) != TreeNode:
        return None
    currentSum = 0
    L = []
    FindPathRecursion(TNode, L, expectedSum, currentSum)


def FindPathRecursion(TNode, L, expectedSum, currentSum):
    # currentSum是不可修改对象，所以不担心冲突
    currentSum += TNode.e
    L.append(TNode.e)
    isleaf = TNode.left == None and TNode.right == None
    if currentSum == expectedSum and isleaf:
        for i in L:
            print(i, end=' ')
        print('\n')

    if TNode.left != None:
        FindPathRecursion(TNode.left, L, expectedSum, currentSum)
    if TNode.right != None:
        FindPathRecursion(TNode.right, L, expectedSum, currentSum)
    L.pop()


# T1=TreeNode(10)
# T2=TreeNode(5)
# T3=TreeNode(12)
# T4=TreeNode(4)
# T5=TreeNode(7)
# T1.left=T2
# T1.right=T3
# T2.left=T4
# T2.right=T5
#
# FindPath(T1,22)


class ComplexListNode:
    def __init__(self, e=None, next=None, Sibling=None):
        self.e = e
        self.next = next
        self.Sibling = Sibling


def CloneNode(Chead):
    if type(Chead) != ComplexListNode:
        return
    CNode = Chead
    while CNode != None:
        Cloned = ComplexListNode()
        Cloned.e = CNode.e
        Cloned.next = CNode.next
        Cloned.Sibling = None
        CNode.next = Cloned
        CNode = Cloned.next


def CloneSibling(Chead):
    CNode = Chead
    while CNode != None:
        Cloned = CNode.next
        if Cloned.Sibling == None:
            Cloned.Sibling = CNode.Sibling.next
        CNode = Cloned.next


def CloneList(Chead):
    CNode = Chead
    ClonedHead = None
    Cloned = None
    if CNode != None:
        ClonedHead = Cloned = CNode.next
        CNode.next = Cloned.next
        CNode = CNode.next
    while CNode:
        Cloned.next = CNode.next
        Cloned = Cloned.next
        CNode.next = Cloned.next
        CNode = CNode.next
    return ClonedHead


def Covert(THead):
    LastNode = None
    LastNode = CovertNode(THead, LastNode)
    phead = LastNode
    while phead and phead.left:
        phead = phead.left
    return phead


def CovertNode(THead, lastNode):
    if THead == None:
        return None
    currentNode = THead
    if currentNode.left != None:
        lastNode = CovertNode(currentNode.left, lastNode)
    currentNode.left = lastNode
    if lastNode:
        lastNode.right = currentNode
    lastNode = currentNode
    if currentNode.right != None:
        lastNode = CovertNode(currentNode, lastNode)
    return lastNode


def Serialize(Thead):
    Serialization = []
    SerializeRecur(Thead, Serialization)
    return Serialization


def SerializeRecur(Thead, L):
    if not Thead:
        L.append('$')
        return
    L.append(Thead.e)
    SerializeRecur(Thead.left, L)
    SerializeRecur(Thead.right, L)


def Permutation(string):
    if not string:
        return
    liststring = list(string)
    PermutationRecur(liststring, 0)


def PermutationRecur(string, start):
    if start >= len(string):
        print(''.join(string))
    else:
        substart = start
        while substart < len(string):
            temp = string[start]
            string[start] = string[substart]
            string[substart] = temp
            PermutationRecur(string, start + 1)
            temp = string[start]
            string[start] = string[substart]
            string[substart] = temp
            substart += 1


# Permutation('ss')


def MoreThanHalf(arr):
    if type(arr) != list:
        return
    if len(arr) == 1:
        return arr[0]
    count = {}
    for i in arr:
        count[i] = count.setdefault(i, 0) + 1
    for key, value in count.items():
        if value > (len(arr) >> 1):
            print('number {} count {} list sum {}'.format(key, value, len(arr)))


# MoreThanHalf([1,1,1,1,1,1,1,1,3,3,3,3,3,3,3])


def MoreThanHalfNoDict(arr):
    if type(arr) != list:
        return
    if len(arr) == 1:
        return arr[0]
    recordedNum = arr[0]
    count = 0
    for i in arr:
        if i == recordedNum:
            count += 1
        else:
            count -= 1
        if count == 0:
            recordedNum = i
    if count:
        print('number {} list sum {}'.format(recordedNum, len(arr)))


# MoreThanHalfNoDict([1,1,1,1,1,1,1,1,3,3,3,3,3,3,3])


def Ngreatest(arr):
    if not arr: return
    currentSum = 0
    GreatestSum = float('-inf')
    i = 0
    while i < len(arr):
        if currentSum <= 0:
            currentSum = arr[i]
        else:
            currentSum += arr[i]
        if currentSum > GreatestSum:
            GreatestSum = currentSum
        i += 1
    return GreatestSum


# print(Ngreatest([1,-2,3,10,-4,7,2,-5]))


def NumberOfOneBetween1AndN(number):
    # 十进制中1的个数
    if type(number) != int or number <= 0:
        return
    strNumber = str(number)
    return NumberOf1(strNumber)


def NumberOf1(strnumber):
    # 十进制数中1的个数
    if not strnumber and strnumber < '0' and strnumber > '9':
        return 0
    first = int(strnumber[0])
    length = len(strnumber)
    if length == 1 and first == 0:
        return 0
    if length == 1 and first > 0:
        return 1
    numberFirstDigit = 0
    if first > 1:
        numberFirstDigit = pow(10, length - 1)
    elif first == 1:
        numberFirstDigit = int(strnumber[1:]) + 1
    numberOtherDigit = first * (length - 1) * pow(10, length - 2)
    numberRecursive = NumberOf1(strnumber[1:])
    return numberFirstDigit + numberOtherDigit + numberRecursive


print(NumberOfOneBetween1AndN(1))


def DigitAtIndex(index):
    if type(index) != int or index < 0:
        return
    digit = 1
    while True:
        numbers = CountOfIntegers(digit)
        if numbers > index:
            return numberAtdigit(index, digit)
        index -= digit * numbers
        digit += 1


def CountOfIntegers(digit):
    if digit == 1:
        return 10
    count = pow(10, digit - 1)
    return 9 * count


def numberAtdigit(index, digit):
    number = BeginNumber(digit) + index // digit
    indexFromRight = digit - index % digit
    i = 1
    while i < indexFromRight:
        number = number // 10
        i += 1
    return number % 10


def BeginNumber(digit):
    if digit == 1:
        return 0
    return pow(10, digit - 1)


# print(DigitAtIndex(1001))

def cmp(a, b):
    print('a=' + a)
    print('b=' + b)
    if a + b > b + a:
        return 1
    if a + b < b + a:
        return -1
    else:
        return 0


def PrintMinNumber(numbers):
    # 连起来最小的整数
    if not numbers:
        return
    number = list(map(str, numbers))
    print(number)
    number.sort(key=cmp_to_key(cmp))
    print(number)
    return ''.join(number).lstrip('0') or '0'


# print(PrintMinNumber([32,3,321]))


def GetTranslation(number):
    if type(number) != int or number < 0:
        return
    stringnumber = str(number)
    return GetTranslationRecur(stringnumber)


def GetTranslationRecur(stringnumber):
    length = len(stringnumber)
    counts = [0] * length
    for i in range(length - 1, -1, -1):
        count = 0
        if i < length - 1:
            count = counts[i + 1]
        else:
            count = 1
        if i < length - 1:
            digit1 = int(stringnumber[i])
            digit2 = int(stringnumber[i + 1])
            converted = digit1 * 10 + digit2
            if converted >= 10 and converted <= 25:
                if i < length - 2:
                    count += counts[i + 2]
                else:
                    count += 1
        counts[i] = count
    return counts[0]


# print(GetTranslation(12258))


def MaxValueOfGift(arr):
    if not arr.size:
        return None
    rows, cols = arr.shape
    maxNumber = [[0] * cols for i in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if i == 0 and j == 0:
                maxNumber[i][j] = arr[i][j]
            elif i - 1 < 0:
                maxNumber[i][j] = arr[i][j] + maxNumber[i][j - 1]
            elif j - 1 < 0:
                maxNumber[i][j] = arr[i][j] + maxNumber[i - 1][j]
            else:
                maxNumber[i][j] = arr[i][j] + max(maxNumber[i - 1][j], maxNumber[i][j - 1])
    return maxNumber[-1][-1]


# gift=np.array([[1,10,3,8],[12,2,9,6],[5,7,4,11],[3,7,16,5]])
# print(MaxValueOfGift(gift))


def LongestSubstringWithoutDuplication(string):
    # O(n)
    if type(string) != str:
        return
    maxlength = 0
    currentlength = 0
    position = {}
    for i in 'abcdefghijklmnopqrstuvwxyz':
        position[i] = -1
    for i in range(len(string)):
        preIndex = position[string[i]]
        if preIndex < 0 or i - preIndex > currentlength:
            currentlength += 1
        else:
            if currentlength > maxlength:
                maxlength = currentlength
            currentlength = i - preIndex
        position[string[i]] = i
    if currentlength > maxlength:
        maxlength = currentlength
    return maxlength


# print(LongestSubstringWithoutDuplication('arabcacfr'))


def MineLongestSubstringWithoutDuplication(string):
    # O(n**2)
    if type(string) != str:
        return
    length = len(string)
    LengthSubstring = [1] * length
    for i in range(length):
        if i == 0:
            LengthSubstring[i] = 1
        else:
            preLength = LengthSubstring[i - 1]
            for j in range(1, preLength + 1):
                if string[i - j] == string[i]:
                    break
                else:
                    LengthSubstring[i] += 1
    return max(LengthSubstring)


# print(MineLongestSubstringWithoutDuplication('arabcacfr'))


def GetUglyNumber(n):
    if type(n) != int or n <= 0:
        return 0
    uglyNumber = [1]
    M2 = 0
    M3 = 0
    M5 = 0
    while len(uglyNumber) < n:
        nextUglyNumber = min(uglyNumber[M2] * 2, uglyNumber[M3] * 3, uglyNumber[M5] * 5)
        uglyNumber.append(nextUglyNumber)
        while uglyNumber[M2] * 2 <= nextUglyNumber:
            M2 += 1
        while uglyNumber[M3] * 3 <= nextUglyNumber:
            M3 += 1
        while uglyNumber[M5] * 5 <= nextUglyNumber:
            M5 += 1

    return uglyNumber[-1]


# print(GetUglyNumber(4))


def InversePair(arr):
    if type(arr) != list or len(arr) == 0:
        return
    copy = [0] * len(arr)
    for i in range(len(arr)):
        copy[i] = arr[i]
    count = InversePairCore(arr, copy, 0, len(arr) - 1)
    return count


def InversePairCore(arr, copy, start, end):
    if start == end:
        return 0
    length = (end - start) // 2
    left = InversePairCore(copy, arr, start, start + length)
    right = InversePairCore(copy, arr, start + length + 1, end)

    i = start + length
    j = end
    indexCopy = end
    count = 0
    while i >= start and j >= start + length + 1:
        if arr[i] > arr[j]:
            copy[indexCopy] = arr[i]
            count += j - start - length
            i -= 1
        else:
            copy[indexCopy] = arr[j]
            j -= 1
        indexCopy -= 1
    while i >= start:
        copy[indexCopy] = arr[i]
        i -= 1
    while j >= start + length + 1:
        copy[indexCopy] = arr[j]
        indexCopy -= 1
        j -= 1
    return left + right + count


# print(InversePair([7,5]))


def FindFirstCommandNode(LNode1, LNode2):
    if type(LNode1) != Node or type(LNode2) != Node:
        return None
    if LNode1 is None or LNode2 is None:
        return 0
    L1 = []
    L2 = []
    while LNode1 != None:
        L1.append(LNode1)
        LNode1 = LNode1._next
    while LNode2 != None:
        L2.append(LNode2)
        LNode2 = LNode2._next
    lastSame = None
    while L1 and L2:
        a1 = L1.pop()
        a2 = L2.pop()
        if a1 != a2:
            return lastSame
        else:
            lastSame = a1
    print(2)
    return lastSame


# T7=Node(7,None)
# T6=Node(6,T7)
# T5=Node(5,T6)
# T4=Node(4,T5)
# T3=Node(3,T6)
# T2=Node(2,T3)
# T1=Node(1,T2)
# print(FindFirstCommandNode(T1,T4))


def GetFirstK(data, k, start, end):
    if start > end:
        return -1
    middleIndex = (start + end) // 2
    middleData = data[middleIndex]
    if middleData == k:
        if (middleIndex > 0 and data[middleIndex - 1] != k) or middleIndex == 0:
            return middleIndex
        else:
            end = middleIndex - 1
    elif middleData > k:
        end = middleIndex - 1
    else:
        start = middleIndex + 1
    return GetFirstK(data, k, start, end)


def GetLastK(data, k, start, end):
    if start > end:
        return -1
    middleIndex = (start + end) // 2
    middleData = data[middleIndex]
    if middleData == k:
        if (middleIndex < len(data) - 1 and data[middleIndex + 1] != k) or middleIndex == len(data) - 1:
            return middleIndex
        else:
            start = middleIndex + 1
    elif middleData < k:
        start = middleIndex + 1
    else:
        end = middleIndex - 1
    return GetLastK(data, k, start, end)


def GetNumberOfK(data, k):
    number = 0
    if type(data) == list and len(data) != 0:
        first = GetFirstK(data, k, 0, len(data) - 1)
        last = GetLastK(data, k, 0, len(data) - 1)
        if first > -1 and last > -1:
            number = last - first + 1
    return number


# print(GetNumberOfK([1,2,3,3,3,3,4,5],3))


def GetMIssingNumber(data):
    if type(data) != list or len(data) == 0:
        return -1
    left = 0
    right = len(data) - 1
    while left <= right:
        middle = (right + left) >> 1
        if data[middle] != middle:
            if middle == 0 or data[middle - 1] == middle - 1:
                return middle
            right = middle - 1
        else:
            left = middle + 1
    if left == len(data):
        return left
    return -1


def GetNumberSameAsIndex(data):
    if type(data) != list or len(data) == 0:
        return -1
    left = 0
    right = len(data) - 1
    while left <= right:
        middle = (left + right) >> 1
        if data[middle] == middle:
            return middle
        if data[middle] > middle:
            right = middle - 1
        else:
            left = middle + 1
    return -1


def TreeDepth(THead):
    if type(THead) != TreeNode or THead == None:
        return 0
    left = TreeDepth(THead.left)
    right = TreeDepth(THead)
    return 1 + max(left, right)


def IsBalancedTree(TRoot):
    if TRoot == None:
        return True
    left = TreeDepth(TRoot.left)
    right = TreeDepth(TRoot.right)
    if abs(left - right) > 1:
        return False
    return IsBalancedTree(TRoot.left) and IsBalancedTree(TRoot.right)


def IsBalanceTree2(TRoot):
    if TRoot is None:
        return 0, True
    left, leftBalance = IsBalanceTree2(TRoot.left)
    right, rightBalance = IsBalanceTree2(TRoot.right)
    if leftBalance and rightBalance:
        if abs(left - right) <= 1:
            return 1 + max(left, right), True
        else:
            return 1 + max(left, right), False
    return 1 + max(left, right), False


class TreeForBalance:
    def __init__(self, e, left=None, right=None):
        self.e = e
        self.left = left
        self.right = right


# T8=TreeForBalance(7)
# T7=TreeForBalance(7,T8)
# T6=TreeForBalance(6)
# T5=TreeForBalance(5,T7)
# T4=TreeForBalance(4)
# T3=TreeForBalance(3,right=T6)
# T2=TreeForBalance(2,T4,T5)
# T1=TreeForBalance(1,T2,T3)
# print(IsBalanceTree2(T1))


def FindNumberAppearOnce(data):
    if type(data) != list or len(data) < 2:
        return
    ResultExclusiveXOR = 0
    for i in data:
        ResultExclusiveXOR ^= i
    num1 = 0
    num2 = 0
    for i in data:
        if i & ResultExclusiveXOR:
            num1 ^= i
        else:
            num2 ^= i
    return num1, num2


# print(FindNumberAppearOnce([2,4,3,6,3,2,5,5]))


def FingTwonumberEqualSum(data, s):
    if type(data) != list or len(data) < 1 or type(s) != int:
        return None
    ahead = 0
    behind = len(data) - 1
    while ahead < behind:
        cursum = data[ahead] + data[behind]
        if cursum == s:
            return data[ahead], data[behind]
        elif cursum < s:
            ahead += 1
        else:
            behind -= 1
    return None


# print(FingTwonumberEqualSum([1,2,4,7,11,15],15))


def findContinousSequence(s):
    if s < 3:
        return
    small = 1
    big = 2
    middle = (1 + s) // 2
    curSum = small + big
    while small < middle:
        if curSum == s:
            for i in range(small, big + 1):
                print(i, end='')
            print()
        while curSum > s and small < middle:
            curSum -= small
            small += 1
            if curSum == s:
                for i in range(small, big + 1):
                    print(i, end='')
                print()
        big += 1
        curSum += big


# findContinousSequence(15)


def Reverse(stringArr, begin, end):
    if not stringArr:
        return
    while begin < end:
        stringArr[begin], stringArr[end] = stringArr[end], stringArr[begin]
        end -= 1
        begin += 1


def ReverseSentence(string):
    if not string:
        return
    stringArr = list(string)
    length = len(stringArr)
    begin = 0
    end = length - 1
    Reverse(stringArr, begin, end)
    begin = end = 0
    while begin < length:
        if stringArr[begin] == ' ':
            begin += 1
            end += 1
        elif end >= length or stringArr[end] == ' ':
            Reverse(stringArr, begin, end - 1)
            begin = end
        else:
            end += 1
    return ''.join(stringArr)


# print(ReverseSentence('I am a student.'))


def PrintProbility(n, side):
    if type(n) != int or n <= 0:
        return
    maxnumber = side
    probility = [0] * (maxnumber * n - n + 1)
    for i in range(1, maxnumber + 1):
        Probility(n, maxnumber, n, i, probility)

    for i in range(len(probility)):
        print('value {} count {} Probility {}'.format(i + n, probility[i], probility[i] / (power(maxnumber, n))))


def Probility(n, maxnumber, current, sum, probility):
    if current == 1:
        probility[sum - n] += 1
    else:
        for i in range(1, maxnumber + 1):
            Probility(n, maxnumber, current - 1, i + sum, probility)


# PrintProbility(6,6)


def PrintProbility2(n, sides):
    if n < 1 or sides <= 1:
        return
    probility = [[], []]
    probility[0] = [0] * (sides * n + 1)
    probility[1] = [0] * (sides * n + 1)
    flag = 0
    for i in range(1, sides + 1):
        probility[flag][i] = 1
    k = 2
    while k <= n:
        for i in range(k):
            probility[1 - flag][i] = 0
        for i in range(k, sides * k + 1):
            probility[1 - flag][i] = 0
            j = 1
            while j <= i and j <= sides:
                probility[1 - flag][i] += probility[flag][i - j]
                j += 1
        k += 1
        flag = 1 - flag
    i = n
    while i <= sides * n:
        print('value {} count {} Probility {}'.format(i, probility[flag][i], probility[flag][i] / power(sides, n)))
        i += 1


# PrintProbility2(6, 6)


def IsContinous(data: list):
    if len(data) != 5:
        return False
    data.sort()
    numberofzero = 0
    numberofGap = 0
    for i in data:
        if i == 0:
            numberofzero += 1
    small = numberofzero
    big = numberofzero + 1
    while big < len(data):
        if data[big] == data[small]:
            return False
        numberofGap += data[big] - data[small] - 1
        small = big
        big += 1
    return False if numberofGap > numberofzero else True


# print(IsContinous([0,2,3,6,7]))


def lastremaining(circle, m):
    if type(circle) != CQueue:
        return None
    if m < 0:
        return None
    while len(circle) != 1:
        k = 1
        while k < m:
            circle.rotate()
            k += 1
        circle.dequeue()
    return circle.first()


C = CQueue()
C.enqueue(0)
C.enqueue(1)
C.enqueue(2)
C.enqueue(3)
C.enqueue(4)


# print(lastremaining(C,3))


def LaseRemaining(n, m):
    if n < 0 or m < 0:
        return None
    last = 0
    i = 2
    while i <= n:
        last = (last + m) % i
    return last


def Maxdiff(data: list):
    if len(data) < 2:
        return None
    minprice = data[0]
    maxdiff = data[1] - minprice
    for i in range(2, len(data)):
        if data[i] < minprice:
            minprice = data[i]
        if data[i] - minprice > maxdiff:
            maxdiff = data[i] - minprice
    return maxdiff


def multiply(data: list):
    if len(data) < 2:
        return None
    C = [1]
    sum = 1
    for i in range(len(data) - 1):
        sum = sum * data[i]
        C.append(sum)
    D = [1]
    sum = 1
    for i in range(len(data) - 1, 0, -1):
        sum = sum * data[i]
        D.append(sum)
    D.reverse()
    B = []
    for i, j in zip(C, D):
        B.append(i * j)

    return B


# print(multiply([1,2,3,4,5]))


def turnstrtonum(string):
    if string:
        number = 0
        i = 0
        minus = False
        while i < len(string):
            if i == 0:
                if string[i] in '+-':
                    if len(string) == 1:
                        return None
                    if string[i] == '-':
                        minus = True
                        i += 1
                    elif string[i] == '+':
                        i += 1
            if string[i] not in '0123456789':
                return None
            number = 10 * number + int(string[i])
            i += 1
        if minus:
            number = -number
        return number
    return None


print(turnstrtonum('0'))
