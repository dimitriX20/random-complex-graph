class dsu(object):
    def __init__(self, n):
        self.parents = [i for i in range(n)]
        self.size = [1] * n  # initialize the size of each element to 1
    
    def find(self, x):
        if self.parents[x] != x:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]
    
    def union(self, x, y):
        rootA, rootB = self.find(x), self.find(y)
        if rootA != rootB:
            # merge the smaller set into the larger set
            if self.size[rootA] < self.size[rootB]:
                self.parents[rootA] = rootB
                self.size[rootB] += self.size[rootA]
            else:
                self.parents[rootB] = rootA
                self.size[rootA] += self.size[rootB]
            return True
        return False 

    def sz(self, node):
        root = self.find(node)
        return self.size[root]
