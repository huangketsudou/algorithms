#-----------------------------------------------------------------------------
#阶乘函数


def factorial(n):
    if n==0:
        return 1
    else:return n*factorial(n-1)



#-----------------------------------------------------------------------------
#英国标尺


def draw_line(tick_length,tick_label=''):
    line='-'*tick_length
    if tick_label:
        line+=' '+tick_label
    print(line)


def draw_interval(center_length):
    if center_length>0:
        draw_interval(center_length-1)
        draw_line(center_length)
        draw_interval(center_length-1)


def draw_ruler(num_inches,major_length):
    draw_line(major_length,'0')
    for j in range(1,1+num_inches):
        draw_interval(major_length-1)
        draw_line(major_length,str(j))


#-----------------------------------------------------------------------------
#良好的斐波那契数列算法——关键在保存F（n-2）的值


def good_fibonacci(n):
    if n<=1:
        return (n,0)
    else:
        (a,b)=good_fibonacci(n-1)
        return (a+b,a)

# print(good_fibonacci(5))


#-----------------------------------------------------------------------
#线性递归计算列表的和


def linear_sum(S,n):
    if n==0:
        return 0
    else:
        return linear_sum(S,n-1)+S[n-1]


#二路递归计算元素之和
def binary_sum(S,start,stop):
    if start>=stop:
        return 0
    elif start==stop-1:
        return S[start]
    else:
        mid=(start+stop)//2
        return binary_sum(S,start,mid)+binary_sum(S,mid,stop)


#---------------------------------------------------------------------------
#递归逆置序列元素


def reverse(S,start,stop):
    if start <stop-1:
        S[start],S[stop-1]=S[stop-1],S[start]
        reverse(S,start+1,stop-1)


#----------------------------------------------------------------------
#计算幂次


def power(x,n):
    if n==0:
        return 1
    else:
        return x*power(x,n-1)


#另一种方法计算幂次
def power_2(x,n):
    if n==0:
        return 1
    else:
        partial=power(x,n//2)
        result=partial*partial
        if n%2==1:
            result*=x
        return result



#----------------------------------PRACTICE------------------------------
def find_max(arr):
    if len(arr)==1:
        return arr[0]
    return arr[0] if arr[0]>find_max(arr[1:]) else find_max(arr[1:])

# print(find_max([1,14,5,6]))

#-----------------------------------------------------------------------
#调和数


def harmonic_num(n):
    if n==1:
        return 1.0/n
    return 1.0/n+harmonic_num(n-1)

# print(harmonic_num(3))

#---------------------------------------------------------------------
#m*n  recursion
def mMultiflyn(m,n):
    if n==1:
        return m
    else:
        return m+mMultiflyn(m,n-1)

print(mMultiflyn(5,4))

