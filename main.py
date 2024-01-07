import matplotlib.pyplot as plt
import network_library as net_lib
import error_function as err_fun
import activation_function as fun
import time
import dataset as ds

#Caricamento del dataset MNIST
Xtrain, Ytrain, Xtest, Ytest = ds.loadDataset()
print("Le caratteristiche del dataset sono:\r")
print("Dimensione Xtrain:", Xtrain.shape)
print("Dimensione Ytrain:", Ytrain.shape)
print("Dimensione Xtest shape:", Xtest.shape)
print("Dimensione Ytest shape:", Ytest.shape, "\r\r")


print("#######################################################Costruzione della rete###############################################\r\r\r")
#Definizione del numero di neuroni hidden
NUM_HIDDEN_NEURONS=500
#Costruzione della rete. Si passa alla funzione: il numero di input, neuroni interni e il numero di classi
network = net_lib.build_network(Xtrain.shape[0],NUM_HIDDEN_NEURONS,Ytrain.shape[0])
#Stampa delle caratteristiche della rete
net_lib.get_network_structure(network)

#Creare una copia della struttura della rete per fare gli esperimenti
network_1=net_lib.copy_network(network)
net_lib.get_network_structure(network_1)

#Split del training set in 2 parti
indiceTraining, indiceValidation = net_lib.splitTrainingDataset(Ytrain.shape[1],20)
print(indiceTraining, indiceValidation)
XT = Xtrain[:,:indiceTraining]
YT = Ytrain[:,:indiceTraining]
XV = Xtrain[:, indiceTraining:indiceTraining+indiceValidation:1]
YV = Ytrain[:, indiceTraining:indiceTraining+indiceValidation:1]
print("XT shape:", XT.shape)
print("YT shape:", YT.shape)
print("XV shape:", XV.shape)
print("YV shape:", YV.shape)

#Training
funzione_errore=err_fun.crossEntropyWithSoftMax
start_time = time.time()
error_training, error_validation,accuracy_training,accuracy_validation= net_lib.train(network_1,funzione_errore,XT,YT,XV,YV, 500, 0.00001,1,4000)
print("Tempo totale di esecuzione della fase di training: %s secondi" % (time.time() - start_time))

plt.figure()
plt.plot(error_training,'b',label='Errore sul Training set')
plt.plot(error_validation,'r',label='Errore sul Validation set')
plt.legend()
plt.show()

plt.figure()
plt.plot(accuracy_training,'b',label='Accuratezza sul Training set')
plt.plot(accuracy_validation,'r',label='Accuratezza sul Validation set')
plt.legend()
plt.show()

accuracy_testSet=net_lib.accuratezza_rete(network_1,Xtest,Ytest)
print('Accuratezza sul test set:', accuracy_testSet)
accuracy_trainingSet=net_lib.accuratezza_rete(network_1,Xtrain,Ytrain)
print('Accuratezza sul training set:', accuracy_trainingSet)
