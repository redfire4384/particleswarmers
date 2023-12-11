# Faye Lawson

import random
import time

def quicksort(A, p, r):
    #Input: an array A[p...r], partition p, range r
    if p < r:
        q = randomizedPartition(A,p,r)
        quicksort(A, p, q-1)
        quicksort(A, q+1, r)

def partition(A, p, r):
    #Input: an array A[p...r], partition p, range r
    x = A[r]
    i = p-1
    for j in range(p, r):
        if A[j] <= x:
            i+=1
            A[i], A[j] = A[j], A[i]
    A[i+1], A[r] = A[r], A[i+1]
    return i + 1

def randomizedPartition(A, p, r):
    #Input: an array A[p...r], partition p, range r
    i =  random.randrange(p,r)
    A[i], A[r] = A[r], A[i]
    return partition(A, p, r)

def testQuick():
    # numlist = [2,8,7,1,3,5,6,4]
    numlist = random.sample(range(-100000,100000), 100000)
    start = time.time()
    quicksort(numlist, 0, len(numlist)-1)
    end = time.time()
    print(f"\nExecution time of quicksort: {(end-start)*10**3:.03f} milliseconds")

testQuick()