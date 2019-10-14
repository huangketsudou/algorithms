from copy import deepcopy
class Empty(Exception):
    pass


class LinkedStack:

    class _Node:
        __slots__ = '_element','_next'

        def __init__(self,element,next):
            self._element=element
            self._next=next


    def __init__(self):
        self._head=None
        self._size=0


    def __len__(self):
        return self._size


    def is_empty(self):
        return self._size==0


    def push(self,e):
        self._head=self._Node(e,self._head)
        self._size+=1


    def top(self):
        if self.is_empty():
            raise Empty("Stack is empty")
        return self._head._element


    def pop(self):
        if self.is_empty():
            raise Empty("Stack is empty")
        answer=self._head._element
        self._head=self._head._next
        self._size-=1
        return answer


def countLink(l):
    if l._next==None:
        return 1
    else:
        return 1+countLink(l._next)


k=LinkedStack()
k.push(1)
k.push(2)
k.push(3)
k.push(4)
k.push(5)

print(countLink(k._head))

