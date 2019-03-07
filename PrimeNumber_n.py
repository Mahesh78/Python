
def prime(n):
    l = []
    if n < 1 :
        print('Enter correct number and try again')
    for i in range(2,n+1):
        for j in range(2,round(i/2)+1):
            if i%j == 0:
                break
        else:
            l.append(i)
    return l,sum(l)