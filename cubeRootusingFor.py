# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 22:48:36 2019

@author: mitikirim
"""


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