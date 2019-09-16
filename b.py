import string
from itertools import permutations


def f(input):
    parsed = input.split("\n")
    T = int(parsed[0])
    ###Loop for all test cases
    for z in range(T):
        counter = 0
        L = int(parsed[z*3+1])
        A = str(parsed[z*3+2])
        B = str(parsed[z*3+3])
        
        
        ###Data manipulation to follow
        maxnum = 1
        for a in range(L):
            maxnum = maxnum*(a+1)
        
        
        ###Test with each letter as the starting letter
        ###soi is the string of interest
        occurence = 0
        for i in range(L):
            flag = 0
            for j in range(i+1,L+1):
                Asubstr = A[i:j]
                #print(Asubstr,'i=',i,'j=',j)
                for substr in set(permutations(Asubstr)):
                    #print(''.join(substr))
                    if B.count(''.join(substr)):
                        occurence += 1
                        break
                    else:
                        flag = 1
                if (flag == 1):
                    break
        print("Case #",z+1,": ",occurence)
f("6\n3\nABB\nBAB\n3\nBAB\nABB\n6\nCATYYY\nXXXTAC\n9\nSUBXXXXXX\nSUBBUSUSB\n4\nAAAA\nAAAA\n19\nPLEASEHELPIMTRAPPED\nINAKICKSTARTFACTORY")