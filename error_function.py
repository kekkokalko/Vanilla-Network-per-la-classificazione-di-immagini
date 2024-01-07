#In questo file sono definite le due funzioni d'errore da sfruttare secondo la traccia
import numpy as np

def sumOfSquares(y,t,flag):
    if(flag==0):
        return (1/2)*np.sum(np.power((y-t),2))
    else:
        return y-t

def crossEntropyWithSoftMax(y,t,flag):
    z=softMax(y)
    if flag==0:
        return -(t*np.log(z)).sum()
    else:
        return z-t

def softMax(y):
    esponenziale=np.exp(y)
    z=esponenziale/sum(esponenziale)
    return z