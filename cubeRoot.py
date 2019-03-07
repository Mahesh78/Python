
def cubeRoot(n):
    x = 1
    while (x**3 < abs(n)):
        x += 1
    if x**3 != abs(n):
        print(str(n) + ' is not a perfect cube')
    else:
        if n < 0:
            x = - x
        print(str(x) + ' is cube root of ' + str(n))
