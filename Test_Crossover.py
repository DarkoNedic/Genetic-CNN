import numpy as np
import random

def crossover(ind1, ind2, BITS_INDICES, q_cpb = 0.3):
	size = min(len(ind1), len(ind2))

	for x in BITS_INDICES:
		if (random.random() < q_cpb):
			#a, b = random.sample(range(size), 2)
			a, b = x[0], x[1]-1
			if a > b:
				a, b = b, a

			print(a,b)

			holes1, holes2 = [True]*size, [True]*size
			for i in range(size):
				if i < a or i > b:
					holes1[ind2[i]] = False
					holes2[ind1[i]] = False
	
			# We must keep the original values somewhere before scrambling everything
			temp1, temp2 = ind1, ind2
			k1 , k2 = b + 1, b + 1
			for i in range(size):
				if not holes1[temp1[(i + b + 1) % size]]:
					ind1[k1 % size] = temp1[(i + b + 1) % size]
					k1 += 1
		
				if not holes2[temp2[(i + b + 1) % size]]:
					ind2[k2 % size] = temp2[(i + b + 1) % size]
					k2 += 1
	
			# Swap the content between a and b (included)
			for i in range(a, b + 1):
				ind1[i], ind2[i] = ind2[i], ind1[i]
	
	return ind1, ind2

NUM_NODES = np.array([3,4,5]) # K
L = 0  # genome length
BITS_INDICES = np.empty((0,2),dtype = np.int32)
start = 0
end = 0
for x in NUM_NODES:
    end = end + sum(range(x))
    BITS_INDICES = np.vstack([BITS_INDICES,[start, end]])
    start = end
L = end
print(BITS_INDICES)

ind1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
ind2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

print(ind1)
print(ind2)
ind11, ind22 = crossover(ind1, ind2, BITS_INDICES)
print(ind11)
print(ind22)
