
def cubeRoot(n):
    for i in range(abs(n)):
        if i**3 == abs(n):
            if n < 0:
                i = -i
            print(str(i)+ ' is cube root of ' + str(n))
            break;
        elif i ** 3 > abs(n):
            print(str(n)+ ' is not a perfect cube')
            break;
