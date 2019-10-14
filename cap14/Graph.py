from copy import deepcopy
from Priorityqueue import AdaptabaleHeapPriorityQueue,HeapPriorityQueue


class Graph:


    class Vertes:
        __slots__ = '_element'
        def __init__(self,x):
            self._element=x


        def element(self):
            return self._element


        def __hash__(self):
            return hash(id(self))


    class Edge:
        __slots__ = '_origin','_destination','_element'
        def __init__(self,u,v,x):
            self._origin=u
            self._destination=v
            self._element=x


        def endpoints(self):
            return (self._origin,self._destination)


        def opposite(self,v):
            return self._destination if v is self._origin else self._origin


        def element(self):
            return self._element


        def __hash__(self):
            return hash((self._origin,self._destination))


    def __init__(self,directed=False):
        self._outgoing={}
        self._incoming={} if directed else self._outgoing


    def is_directed(self):
        return self._incoming is not self._outgoing


    def vertex_count(self):
        return len(self._outgoing)


    def vertices(self):
        return self._outgoing.keys()


    def edge_count(self):
        total=sum(len(self._outgoing[v]) for v in self._outgoing)
        return total if self.is_directed() else  total//2


    def edges(self):
        result=set()
        for secondary_map in self._outgoing.values():
            result.update(secondary_map.values())
        return result


    def get_edge(self,u,v):
        return self._outgoing[u].get(v)


    def degree(self,v,outgoing=True):
        adj=self._outgoing if outgoing else self._incoming
        return len(adj[v])


    def incident_edges(self,v,outgoing=True):
        adj=self._outgoing if outgoing else self._incoming
        for edge in adj[v].values():
            yield edge


    def insert_vertex(self,x=None):
        v=self.Vertes(x)
        self._outgoing[v]={}
        if self.is_directed():
            self._incoming[v]={}


    def insert_edge(self,u,v,x=None):
        e=self.Edge(u,v,x)
        self._outgoing[u][v]=e
        self._incoming[v][u]=e


def DFS(g,u,discovered):
    for e in g.incident_edges(u):
        v=e.opposite(u)
        if v not in discovered:
            discovered[v]=e
            DFS(g,v,discovered)


def construct_path(u,v,discovered):
    path=[]
    if v in discovered:
        path.append(v)
        walk=v
        while walk is not u:
            e=discovered[walk]
            parent=e.opposite(walk)
            path.append(parent)
            walk=parent
        path.reverse()
    return path


def DFS_complete(g):
    forest={}
    for u in g.vertices():
        if u not in forest:
            forest[u]=None
            DFS(g,u,forest)
    return forest


def BFS(g,s,discovered):
    level=[s]
    while len(level)>0:
        next_level=[]
        for u in level:
            for e in g.incident_edges(u):
                v=e.opposite(u)
                if v not in discovered:
                    discovered[v]=e
                    next_level.append(v)
        level=next_level


def floyd_warshall(g):
    closure=deepcopy(g)
    verts=list(closure.vertices())
    n=len(verts)
    for k in range(n):
        for i in range(n):
            if i!=k and closure.get_edge(verts[i],verts[k]) is not None:
                for j in range(n):
                    if i!=k!=j and closure.get_edge(verts[k],verts[j]) is not None:
                        if closure.get_edge(verts[i],verts[j]) is None:
                            closure.insert_edge(verts[i],verts[j])
    return closure


def topological_sort(g):
    #针对有向图
    topo=[]
    ready=[]
    incount={}
    for u in g.vertices():
        incount[u]=g.degree(u,False)
        if incount[u]==0:
            ready.append(u)
    while len(ready)>0:
        u=ready.pop()
        topo.append(u)
        for e in g.incident_edges(u):
            v=e.opposite(u)
            incount[v]-=1
            if incount[v]==0:
                ready.append(v)
    return topo


def shortest_path_lengths(g,src):
    d={}
    cloud={}
    pq=AdaptabaleHeapPriorityQueue()
    pqlocator={}
    for v in g.vertices():
        if v is src:
            d[v]=0
        else:
            d[v]=float('inf')
        pqlocator[v]=pq.add(d[v],v)
    while not pq.is_empty():
        key,u=pq.remove_min()
        cloud[u]=key
        del pqlocator[u]
        for e in g.incident_edges(u):
            v=e.opposite(u)
            if v not in cloud:
                wgt=e.element()
                if d[u]+wgt<d[v]:
                    d[v]=d[u]+wgt
                    pq.update(pqlocator[v],d[v],v)
    return cloud


def shortest_path_tree(g,s,d):
    tree={}
    for v in d:
        if v is not s:
            for e in g.incident_edges(v,False):
                u=e.opposite(v)
                wgt=e.element()
                if d[v]==d[u]+wgt:
                    tree[v]=e
    return tree


def MST_PrimJarnik(g):
    d={}
    tree=[]
    pq=AdaptabaleHeapPriorityQueue()
    pqlocator={}
    for v in g.vertices():
        if len(d)==0:
            d[v]=0
        else:
            d[v]=float('inf')
        pqlocator[v]=pq.add(d[v],(v,None))
    while not pq.is_empty():
        key,value=pq.remove_min()
        u,edge=value
        del pqlocator[u]
        if edge is not None:
            tree.append(edge)
        for link in g.incident_edges(u):
            v=link.opposite(u)
            if v in pqlocator:
                wgt=link.element()
                if wgt<d[v]:
                    d[v]=wgt
                    pq.update(pqlocator[v],d[v],(v,link))
        return tree


def MST_Kruskal(g):
    tree=[]
    pq=HeapPriorityQueue()
    forest=Partition()
    position={}

    for v in g.vertices():
        position[v]=forest.make_group(v)

    for e in g.edges():
        pq.add(e.element(),e)

    size=g.vertex_count()
    while len(tree)!=size-1 and not pq.is_empty():
        weight,edge=pq.remove_min()
        u,v=edge.endpoints()
        a=forest.find(position[u])
        b=forest.find(position[v])
        if a!=b:
            tree.append(edge)
            forest.union(a,b)

    return tree


class Partition:
    class Position:
        __slots__ = '_container','_element','_size','_parent'


        def __init__(self,container,e):
            self._container=container
            self._element=e
            self._size=1
            self._parent=self


        def element(self):
            return self._element


    def find(self,p):
        if p._parent!=p:
            p._parent=self.find(p._parent)
        return p._parent


    def union(self,p,q):
        a=self.find(p)
        b=self.find(q)
        if a is not b:
            if a._size>b._size:
                b._parent=a
                a._size+=b._size
            else:
                a._parent=b
                b._size+=a._size


    def make_group(self,e):
        return self.Position(self,e)


