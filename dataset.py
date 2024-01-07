import numpy as np
import matplotlib.pyplot as plt

def loadDataset(data_path = './MNIST/'):
    image_size=28
    classes=10

    train_data=np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
    test_data=np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

    #creazione delle 4 matrici xtrain, ytrain, xtest, ytest
    xtrain=np.asarray(train_data[:,1:])
    ytrain=np.asarray(train_data[:,:1])
    xtest=np.asarray(test_data[:,1:])
    ytest=np.asarray(test_data[:,:1])

    #trasformazione delle etichette in rappresentazione one-hot
    range=np.arange(classes) #[0 1 2 3 4 5 6 7 8 9]
    #Se la classe di appartenenza è stata incontrata, poni 1 nella lista,altrimenti 0
    ytrain = (range==ytrain).astype(int)
    ytest = (range==ytest).astype(int)

    return xtrain.transpose(), ytrain.transpose(), xtest.transpose(), ytest.transpose()

#Funzione per la visualizzazione di un'immagine selezionata
def showImage(immagine):
    Immagine=immagine.reshape[(28,28)]
    plt.imshow(Immagine,'gray')
    plt.show()