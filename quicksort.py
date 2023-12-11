import random

def quicksort(A, p, r):
    #Input: an array A[p...r], partition p, range r
    if p < r:
        q = randomizedPartition(A,p,r)
        quicksort(A, p, q-1)
        quicksort(A, q+1, r)

def partition(A, p, r):
    #Input: an array A[p...r], partition p, range r
    x = A[r][0]
    i = p-1
    for j in range(p, r):
        if A[j][0] <= x:
            i+=1
            A[i][0], A[j][0] = A[j][0], A[i][0]
    A[i+1][0], A[r][0] = A[r][0], A[i+1][0]
    return i + 1

def randomizedPartition(A, p, r):
    #Input: an array A[p...r], partition p, range r
    i =  random.randrange(p,r)
    A[i][0], A[r][0] = A[r][0], A[i][0]
    return partition(A, p, r)