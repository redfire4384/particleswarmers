# Faye Lawson

import random

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
