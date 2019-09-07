import numpy as np

class AllIndividuals:
    def __init__(self):
        self.individuals = []

    def add_ind(self, ind, fitness):
        self.individuals.append((fitness, ind))
    def top(self, k):
        top = np.asarray(self.individuals)
        return sorted(self.individuals, reverse=True, key=lambda x: x[0])[:k]

BI = AllIndividuals()
BI.add_ind([1,2,3,4], 0.99)
BI.add_ind([5,6,7,8], 0.33)
BI.add_ind([5,6], 0.43)
toplist = BI.top(2)
for x in toplist:
    print(x[1])
    print("Fitness score:", x[0], "\n")
