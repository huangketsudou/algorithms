class Empty(Exception):
    pass


class _DoublyLinkedBase:

    class _Node:
        __slots__ = '_element','_prev','_next'
        def __init__(self,element,prev,next):
            self._element=element
            self._prev=prev
            self._next=next

    def __init__(self):
        self._header=self._Node(None,None,None)
        self._trailer=self._Node(None,None,None)
        self._header._next=self._trailer
        self._trailer._prev=self._header
        self._size=0


    def __len__(self):
        return self._size


    def is_empty(self):
        return self._size==0


    def _insert_between(self,e,predecessor,successor):
        newest=self._Node(e,predecessor,successor)
        predecessor._next=newest
        successor._prev=newest
        self._size+=1
        return newest


    def _delete_node(self,node):
        predecessor=node._prev
        successor=node._next
        predecessor._next=successor
        successor._prev=predecessor
        self._size-=1
        element=node._element
        node._prev=node._next=node._element=None
        return element


    def __reversed__(self):
        start=self._header
        start_next=self._header._next
        while start._next:
            start_next=start_next._next
            start._prev,start._next=start._next,start._prev
            start=start_next
        start._prev,start._next=start._next,start._prev
        self._header,self._trailer=self._trailer,self._header


class LinkedDeque(_DoublyLinkedBase):

    def first(self):
        if self.is_empty():
            raise Empty("Deque is empty")
        return self._header._next._element


    def last(self):
        if self.is_empty():
            raise Empty('Deque is empty')
        return self._trailer._prev._element


    def insert_first(self,e):
        self._insert_between(e,self._header,self._header._next)


    def insert_last(self,e):
        self._insert_between(e,self._trailer._prev,self._trailer)


    def delete_first(self):
        if self.is_empty():
            raise Empty('Deque is empty')
        return self._delete_node(self._header._next)


    def delete_last(self):
        if self.is_empty():
            raise Empty('Deque is empty')
        return self._delete_node(self._trailer._prev)


    def reverse(self):
        self.__reversed__()


def searchMedium(L):
    start=L._header._next
    end=L._trailer._prev
    while not (start==end or start._next==end):
        start=start._next
        end=end._prev
    return start

# g=LinkedDeque()
# g.insert_first(5)
# g.insert_last(2)
# g.insert_first(3)
# g.insert_last(4)
# g.reverse()
# print(g.last())
# print(g.first())

class PositionList(_DoublyLinkedBase):

    class Position:

        def __init__(self,container,node):
            self._container=container
            self._node=node


        def element(self):
            return self._node._element


        def __eq__(self, other):
            return type(other) is type(self) and other._node is self._node

        def __ne__(self, other):
            return not (self==other)

    def _validate(self,p):
        if not isinstance(p,self.Position):
            raise TypeError('p must be proper Position type')
        if p._container is not self:
            #与_make_position有关
            raise ValueError('p does not belong to this container')
        if p._node._next is None:
            raise ValueError('p is no longer valid')
        return p._node


    def _make_position(self,node):
        if node is self._header or node is self._trailer:
            return None
        else:
            return self.Position(self,node)


    def first(self):
        return self._make_position(self._header._next)


    def last(self):
        return self._make_position(self._trailer._prev)


    def before(self,p):
        node=self._validate(p)
        return self._make_position(node._prev)


    def after(self,p):
        node=self._validate(p)
        return self._make_position(node._next)


    def __iter__(self):
        cursor=self.first()
        while cursor is not None:
            yield cursor.element()
            cursor=self.after(cursor)


    def __reversed__(self):
        cursor=self.last()
        while cursor is not None:
            yield cursor.element()
            cursor=self.before(cursor)


    def _insert_between(self,e,predecessor,successor):
        node=super()._insert_between(e,predecessor,successor)
        return self._make_position(node)


    def add_first(self,e):
        return self._insert_between(e,self._header,self._header._next)


    def add_last(self,e):
        return self._insert_between(e,self._trailer._prev,self._trailer)


    def add_before(self,p,e):
        original=self._validate(p)
        return self._insert_between(e,original._prev,original)


    def add_after(self,p,e):
        original=self._validate(p)
        return self._insert_between(e,original,original._next)


    def delete(self,p):
        original=self._validate(p)
        return self._delete_node(original)


    def replace(self,p,e):
        original=self._validate(p)
        old_value=original._element
        original._element=e
        return old_value


    def swap(self,p,q):
        p_original=self._validate(p)
        q_original=self._validate(q)
        p_original._prev,q_original._prev=q_original._prev,p_original._prev
        p_original._next, q_original._next = q_original._next, p_original._next
        return p_original,q_original



    def max(self):
        maxnum=self.first().element()
        for i in self:
            if i>=maxnum:
                maxnum=i
        return maxnum


    def find(self,e):
        start=self.first()
        for i in self:
            if e==i:
                return start
            else:
                start=self.after(start)
        return None


    def recur_find(self,e):
        start=self.first()
        return self._recur_find(start,e)


    def _recur_find(self,p,e):
        if p==None:
            return None
        elif p.element()==e:
            return p
        else:
            return self._recur_find(self.after(p),e)


def insertion_sort(L):
    if len(L)>1:
        marker=L.first()
        while marker !=L.last():
            pivot=L.after(marker)
            value=pivot.element()
            if value>marker.element():
                marker=pivot
            else:
                walk=marker
                while walk!=L.first() and L.before(walk).element()>value:
                    walk=L.before(walk)
                L.delete(pivot)
                L.add_before(walk,value)


class FavouriteList:
    class _Item:
        __slots__ = '_value','_count'
        def __init__(self,e):
            self._value=e
            self._count=0


    def __init__(self):
        self._data=PositionList()


    def __len__(self):
        return len(self._data)


    def is_empty(self):
        return len(self._data)==0


    def _find_position(self,e):
        walk=self._data.first()
        while walk is not None and walk.element()._value!=e:
            walk=self._data.after(walk)
        return walk


    def _move_up(self,p):
        if p!=self._data.first():
            cnt=p.element()._count
            walk=self._data.before(p)
            if cnt>walk.element()._count:
                while (walk!=self._data.first() and cnt>self._data.before(walk).element()._count):
                    walk=self._data.before(walk)
                self._data.add_before(walk,self._data.delete(p))



    def access(self,e):
        p=self._find_position(e)
        if p is None:
            p=self._data.add_last(self._Item(e))
        p.element()._count+=1
        self._move_up(p)


    def remove(self,e):
        p=self._find_position(e)
        if p is not None:
            self._data.delete(p)


    def clear(self):
        start=self._data.first()._node
        while start._next:
            temp=start._next
            start._prev=start._next=start._element=None
            start=temp
        self._data._header._next=self._data._trailer
        self._data._trailer._prev=self._data._header
        self._data._size=0



    def reset_count(self):
        if self.is_empty():
            return
        for i in self._data:
            i.element()._count=0


    def top(self,k):
        if self.is_empty():
            raise Empty('Empty List')
        if not 1<=k<=len(self):
            raise ValueError('Illegal value for k')
        walk=self._data.first()
        for j in range(k):
            item=walk.element()
            yield item._value
            walk=self._data.after(walk)



class FavouriteListMTF(FavouriteList):
    def _move_up(self,p):
        if p!=self._data.first():
            self._data.add_first(self._data.delete(p))


    def top(self,k):
        if not 1<=k<=len(self):
            raise ValueError("Illegal value for k")
        temp=PositionList()
        for item in self._data:
            temp.add_last(item)
        for j in range(k):
            highPos=temp.first()
            walk=temp.after(highPos)
            while walk is not None:
                if walk.element()._count >highPos.element()._count:
                    highPos=walk
                walk=temp.after(walk)
            yield highPos.element()._value
            temp.delete(highPos)




if __name__=='__main__':


    L=PositionList()
    p=L.add_last(8)
    L.first()
    q=L.add_after(p,5)
    L.before(q)
    r=L.add_before(q,3)
    print(r.element())
    L.after(p)
    print(L.before(p))
    s=L.add_first(10)
    print(L.delete(L.last()))
    L.replace(p,7)
    print(L.max())
    print(L.recur_find(3))
    L.swap(r,p)

    # F=FavouriteList()
    # F.access(1)
    # F.access(1)
    # F.access(3)
    # F.access(2)
    # F.access(2)
    # for i in F.top(2):
    #     print(i)
    # F.clear()