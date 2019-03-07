
def cubeRoot(n):
    t = 0
    for i in range(abs(n)):
        t += 1
        if i**3 == abs(n):
            break
    print(t)
    if i**3 != abs(n):
        print('Not')
    
    else:
        if n < 0:
            i = -i
        print(str(i)+ ' is cube root of ' + str(n))
