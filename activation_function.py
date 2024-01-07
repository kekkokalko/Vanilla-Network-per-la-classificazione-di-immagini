import numpy as np

def sigmoid(x,flag):
    y = 1 / (1 + np.exp(-x))
    if flag==0:
        return y
    else:
        return y, y*(1-y)

def identity(x,flag):
    y=x
    if flag==0:
        return y
    else:
        return y,1

def tanh(x,flag):
    y=(1-np.exp(-2*x))/(1+np.exp(+2*x))
    if(flag==0):
        return y
    else:
        return y, 1-y*y

def relu(x,flag):
    y=(x>0)
    if flag==0:
        return y*x
    else:
        return y*x, y*1