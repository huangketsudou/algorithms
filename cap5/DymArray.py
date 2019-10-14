import ctypes


class DynamicArray:
    def __init__(self):
        self._n=0
        self._capacity=1
        self._A=self._make_array(self._capacity)


    def __len__(self):
        return self._n


    def __getitem__(self, k):
        if k>=0:
            return self._A[k]
        else:
            return self._A[self._n+k]


    def append(self,obj):
        if self._n==self._capacity:
            self._resize(2*self._capacity)
        self._A[self._n]=obj
        self._n+=1


    def insert(self,k,value):
        if self._n==self._capacity:
            self._resize(2*self._capacity)
        for j in range(self._n,k,-1):
            self._A[j]=self._A[j-1]
        self._A[k]=value
        self._n+=1


    def pop(self):
        self._A[-1]=None
        self._n-=1
        if self._n<int(self._capacity/4):
            self._resize(int(self._capacity/2))


    def remove(self,value):
        for k in range(self._n):
            if self._A[k]==value:
                for j in range(k,self._n-1):
                    self._A[j]=self._A[j+1]
                self._A[self._n-1]=None
                self._n-=1
                return
        raise ValueError('value not found')


    def remove_all(self,value):
        tmp=DynamicArray()
        for k in range(self._n):
            if self._A[k]==value:
                pass
            else:
                tmp.append(self._A[k])
        self._A=tmp
        self._n=tmp._n
        self._capacity=tmp._capacity


    def _resize(self,c):
        B=self._make_array(c)
        for k in range(self._n):
            B[k]=self._A[k]
        self._A=B
        self._capacity=c


    def _make_array(self,c):
        return (c*ctypes.py_object)()

a=DynamicArray()
a.append(1)
a.append(2)
a.append(3)
a.append(1)
a.append(1)
a.remove_all(1)
print(len(a))