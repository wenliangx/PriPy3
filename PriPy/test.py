class m:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):
        if self.step < self.n:
            self.step += 1
            return self
        else:
            raise StopIteration()


for i in m(5):
    print(type(i))
    print(i.step)

