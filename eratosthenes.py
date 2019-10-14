def eratosthenes(n):
    IsPrime=[True]*(n+1)
    for i in range(2,int(n**0.5)+1):
        if IsPrime[i]:
            for i in range(i*i,n+1,i):
                IsPrime[i]=False
    Prime=[i for i in range(2,n+1) if IsPrime[i]]
    return Prime



if __name__ == "__main__":
    print(eratosthenes(100))
