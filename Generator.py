import random
import sys

assert len(sys.argv) > 2, "Please supply: file name, number of instances, followed by number of objectives"

f = sys.argv[1]
n = int(sys.argv[2])
m = int(sys.argv[3])

f = open(f,"w+")
f.write(str(n) + " " + str(m) + "\n")

for o in range(m):
    for i in range(n):
        for j in range(i):
            f.write(str(random.randint(1, sys.maxsize)) + "\n")

f.close()