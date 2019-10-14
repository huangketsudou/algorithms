def processing(str):
    return '#'+'#'.join(list(str))+'#'


def find(str,center,length):
    #注意这里的and判定，在编译器中前方判断为False时，后方不会再执行，可以避开outOfRange错误
    while center-length-1>=0  and center+length+1<len(str) and str[center-length-1]==str[center+length+1]:
        length+=1
    return length

def manacher(str):
    str=processing(str)
    rightMax=0
    MaxCenter=0
    MaxLength=0
    length=[0]*len(str)
    for i in range(1,len(str)-1):
        if i>=rightMax:
            length[i]=find(str,i,0)
            MaxCenter=i
            rightMax=MaxCenter+length[i]
            if length[i]>MaxLength:
                MaxLength=length[i]
        else:
            LeftI=2*MaxCenter-i
            if i+length[LeftI]<rightMax:
                length[i]=length[LeftI]
            else:
                length[i]=find(str,i,length[LeftI])
                MaxCenter=i
                rightMax=MaxCenter+length[i]
                if length[i] > MaxLength:
                    MaxLength = length[i]
    return MaxLength








print('-------------------------------')
print('input your string:')
str=str(input())

print(manacher(str))
