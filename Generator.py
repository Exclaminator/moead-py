import random
import sys
import os

# show error on incomplete arguments
assert len(sys.argv) > 2, "Please supply: folder name, maximum number of instances, " \
                          "followed by maximum number of objectives"
# load arguments
instanceName = sys.argv[1]
maxn = int(sys.argv[2])
maxm = int(sys.argv[3])

# Create target Directory if don't exist
if not os.path.exists(instanceName):
    os.mkdir(instanceName)

# for all combinations of n and objectives
for n in range(1, maxn):
    for m in range(1, maxm):

        # make an instance
        f = open(instanceName + "/instance_" + str(n) + "_" + str(m) + ".tsp", "w+")
        f.write(str(n) + " " + str(m) + "\n")

        # only upper trinagle matrix needs to be represented
        for o in range(m):
            for i in range(n):
                for j in range(i):
                    f.write(str(random.randint(1, n*100)) + "\n")

        # close file
        f.close()

    # log our progress
    print("Made all instances up to size " + str(n))
print("Done")
