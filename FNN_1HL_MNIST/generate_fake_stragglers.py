# Standard modules imports
import random
import operator

num_stragglers = 1125
num_batches = 1
batches_lenght = 10000

fake_stragglers = []

for i in range(num_stragglers):
    fake_stragglers += [(random.randint(0, num_batches - 1), random.randint(0, batches_lenght - 1))]
fake_stragglers.sort(key=operator.itemgetter(0,1))

f = open("./stragglers_list/fake_stragglers.dat", 'w+')
f.write('[[\n')
for i in range(len(fake_stragglers)):
    f.write('{},\n'.format(fake_stragglers[i]))
f.write(']]')
f.close()