class Empty(Exception):
    pass


class CircularQueue:
    class _Node:
        __slots__ = '_element', '_next'

        def __init__(self, element, next):
            self._element = element
            self._next = next


    def __init__(self):
        self._tail=None
        self._size=0


    def __len__(self):
        return self._size


    def _is_empty(self):
        return self._size==0


    def first(self):
        if self._is_empty():
            raise Empty('Queue is empty')
        head=self._tail._next
        return head


    def dequeue(self):
        if self._is_empty():
            raise Empty("Queue is empty")
        oldhead=self._tail._next
        if self._size==1:
            self._tail=None
        else:
            self._tail._next=oldhead._next
        self._size-=1
        return oldhead._element


    def enqueue(self,e):
        newest=self._Node(e,None)
        if self._is_empty():
            newest._next=newest
        else:
            newest._next=self._tail._next
            self._tail._next=newest
        self._tail=newest
        self._size+=1


    def rotate(self):
        if self._size>0:
            self._tail=self._tail._next

def count(l):
    start=l._tail
    num=1
    l.rotate()
    while start!=l._tail:
        num+=1
        l.rotate()
    return num


l=CircularQueue()
l.enqueue(1)
l.enqueue(2)
l.enqueue(3)



print(count(l))
