# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 22:26:50 2019

@author: mitikirim
"""


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