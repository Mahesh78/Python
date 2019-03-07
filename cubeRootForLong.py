# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 22:55:52 2019

@author: mitikirim
"""


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