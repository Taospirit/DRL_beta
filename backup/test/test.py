class A:
    def __init__(self, x):
        self.x = x

    def set(self, x):
        self.x = x

class B:
    def __init__(self, x):
         self.b1 = x
         self.b2 = x

    

a = A(1)
b = B(a)
print (b.b1.x)
print (b.b2.x)
a.set(3)
print (b.b1.x)
print (b.b2.x)
