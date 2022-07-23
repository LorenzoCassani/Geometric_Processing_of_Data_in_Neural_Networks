# Standard modules imports
import sys
import operator

union = []

i = 1
while i <= 30:
    f = open('./stragglers_list/stragglers_{}.dat'.format(i), 'r')
    stragglers = eval(f.read())
    f.close()
    for run in range(len(stragglers)):
        union = union + stragglers[run]
    i += 1
stragglers = list(dict.fromkeys(union))
stragglers.sort(key=operator.itemgetter(0,1))

f = open('./stragglers_list/stragglers_merged.dat', 'w+')
f.write('[[\n')
for i in range(len(stragglers)):
    f.write('{},\n'.format(stragglers[i]))
f.write(']]')
f.close()