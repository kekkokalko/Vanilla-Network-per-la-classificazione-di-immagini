import numpy as np
import activation_function as fun
import error_function


#Funzione per la costruzione della rete neurale
def build_network(input_size,num_hidden_neurons,output_size):
    pesi=[]
    bias=[]
    funzione_attivazione=[]
    j=input_size
    hidden_neurons=[num_hidden_neurons]
    #Costruzione degli strati interni della rete
    for i in hidden_neurons:
        #Definizione dei pesi iniziali della rete attraverso una gaussiana con dev.std=0.1 e media nulla
        #Vengono aggiunti alle 2 liste degli array di valori numerici estratti secondo la gaussiana, di i righe e j colonne
        bias.append(np.random.normal(loc=0.0,scale=0.1,size=[i,1]))
        pesi.append(np.random.normal(loc=0.0,scale=0.1,size=[i,j]))
        j=i
        #Definizione della funzione d'attivazione sullo strato hidden: sigmoide
        funzione_attivazione.append(fun.sigmoid)

    #Definizione dello strato d'output con la stessa logica
    pesi.append(np.random.normal(loc=0.0,scale=0.1,size=[output_size, j]))
    bias.append(np.random.normal(loc=0.0,scale=0.1,size=[output_size, 1]))
    funzione_attivazione.append(fun.identity)
    network={'Pesi':pesi,'Bias':bias,'Funzione d attivazione': funzione_attivazione, 'Profondità': len(pesi)}
    return network

#Funzione che restituisce le caratteristiche di una rete datale in input
def get_network_structure(network):
    lunghezza=len(network['Pesi'])
    numero_strati_hidden=lunghezza-1
    input_size= network['Pesi'][0].shape[1]
    output_size= network['Pesi'][numero_strati_hidden].shape[0]
    activation_function=[]
    numero_neuroni_strati_hidden=[]

    for i in range(numero_strati_hidden):
        numero_neuroni_strati_hidden.append(network['Pesi'][i].shape[0])
        activation_function.append(network['Funzione d attivazione'][i].__name__)
    activation_function.append(network['Funzione d attivazione'][numero_strati_hidden].__name__)
    print('Numero di strati hidden: ', numero_strati_hidden)
    print('Input size: ', input_size)
    print('Output size: ', output_size)
    print('Numero di neuroni per ogni strato hidden: ')
    for i in range(len(numero_neuroni_strati_hidden)):
        print(numero_neuroni_strati_hidden[i])
    print('Funzione d ativazione: ')
    for i in range(len(numero_neuroni_strati_hidden)+1):
        print(activation_function[i])
    return

#Funzione che permette di splittare il training set in: 80% training set effettivo e 20% validation set
#Vengono ritornati gli indici, secondo la logica descritta, per ricopiare i valori nei vari insiemi
def splitTrainingDataset(Y, percentuale):
    #Proporzioone => 20:100=x:YTrain.shape
    indiceValidation = int((percentuale*Y)/100)
    #Proporzione => 80:100=x:YTrain.shape
    indiceTraining = int(((100-percentuale)*Y)/100)
    return indiceTraining, indiceValidation

#Funzione che effettua la copia di una rete in un'altra
def copy_network(network):
    Pesi=[]
    Bias=[]
    numero_livelli=len(network['Pesi'])
    for i in range(numero_livelli):
        Pesi.append(network['Pesi'][i].copy())
        Bias.append(network['Bias'][i].copy())
    network_1={'Pesi':Pesi,'Bias':Bias,'Funzione d attivazione': network['Funzione d attivazione'], 'Profondità': numero_livelli}
    return network_1

#Funzione che definisce la fase di learning della rete
#network: la rete da addestrare (Parametro di input e output)
#err_fun: funzione d'errore da usufruire
#XTrain,YTrain,XValid,YValid: training set e validation set (coppie di valori, insieme ai target)
#max_epoche: numero massimo di epoche da usufuire
#eta: iperparametro della rete necessario per definire la discesa del gradiente in fase di aggiornamento dei parametri
#flag: variabile che definisce la modalità di learning da seguire (batch, minibatch o online)
#numero_mini_batch: variabile che definisce il quantitativo di batch da creare nel caso si scelga la modalità minibatch
def train(network, err_fun, XTrain, YTrain, XValid, YValid, maxEpoche,eta,flag,numero_mini_batch):
    profondita=network['Profondità']
    errore_training=[]
    errore_validation=[]
    accuratezza_training=[]
    accuratezza_validation=[]
    if flag==0:
        print('Modalità batch')
        numero_mini_batch=1
        #Training set sarà passato tutto in una volta alla rete
        training_effettivo=split(YTrain,numero_mini_batch)
    elif numero_mini_batch>0:
        print('Modalità mini batch')
        #La rete verrà addestrata su un batch alla volta
        training_effettivo=split(YTrain,numero_mini_batch)
    else:
        print('Modalità online')
        #La rete verrà addestrata su pezzi pari alla quantità dei dati di ogni classe
        training_effettivo=[[i] for i in np.arange(YTrain.shape[1])]
    #Addestramento della rete sulla 1° epoca per ricavare la prima rete migliore
    y_training = forward_propagation(network, XTrain, XTrain,0)
    errore = err_fun(y_training, YTrain, 0)
    errore_training.append(errore)
    y_validation = forward_propagation(network, XValid, XValid,0)
    errore_val = err_fun(y_validation, YValid, 0)
    errore_validation.append(errore_val)
    minimo_errore_validation = errore_val
    rete_migliore = copy_network(network)
    accuratezza_training.append(accuratezza(y_training, YTrain))
    accuratezza_validation.append(accuratezza(y_validation, YValid))
    print('Epoca:', 0,
          'Errore sul training set:', errore_training,
          'Errore sul validation set:', errore_validation,
          'Accuracy sul training set:', accuratezza(y_training, YTrain),
          'Accuracy sul validation set:', accuratezza(y_validation, YValid))

    #Fase di learning effettiva
    for epoca in range(maxEpoche):      #Ciclo con il suo criterio d'arresto
        for i in training_effettivo:    #Si lavora sul quantitativo di dati effettivo, in base alla modalità di apprendimento scelta
            output_strati,derivate_strati=forward_propagation(network,XTrain,XTrain[:,i],1)         #Fase forward che ritorna, a mo' di liste, gli output di ciascun layer e le loro derivate
            derivate_pesi,derivate_bias=back_propagation(network,YTrain[:,i],err_fun,output_strati,derivate_strati)     #Fase di backpropagation per il calcolo di: delta valori e derivate
            for j in range(profondita):             #Fase di aggiornamento dei pesi secondo lo standard gradient descent
                network['Pesi'][j]=network['Pesi'][j]-eta*derivate_pesi[j]
                network['Bias'][j]=network['Bias'][j]-eta*derivate_bias[j]
        #Calcolo dell'errore, tramite error function e conseguete ricerca della rete migliore
        y_training = forward_propagation(network, XTrain, XTrain,0)
        errore = err_fun(y_training, YTrain, 0)
        errore_training.append(errore)
        y_validation = forward_propagation(network, XValid, XValid,0)
        errore_val = err_fun(y_validation, YValid, 0)
        errore_validation.append(errore_val)
        accuratezza_training.append(accuratezza(y_training, YTrain))
        accuratezza_validation.append(accuratezza(y_validation, YValid))
        if errore_val < minimo_errore_validation:
            minimo_errore_validation = errore_val
            rete_migliore = copy_network(network)
        print('Epoca:', epoca+1,
              'Errore sul training set:', errore,
              'Errore sul validation set:', errore_val,
              'Accuracy sul training set:', accuratezza(y_training, YTrain),
              'Accuracy sul validation set:', accuratezza(y_validation, YValid))
        print('\r', end='')

    network=copy_network(rete_migliore)
    return errore_training,errore_validation,accuratezza_training,accuratezza_validation

def split(YTrain,k):
    dimensione_dataset=YTrain.shape[0]
    #Creazione dei k batch (al momento di dimensione pari a 0)
    indiciYTrain = [np.ndarray(0, int) for i in range(k)]
    for i in range(dimensione_dataset): #Scorrimento da 0 a 9 (classi)
        v=(YTrain.argmax(0)==i)         #Se l'elemento, di YTrain, più grande (cioè 1) è in posizione i (cioè la classe corrente), metti true, altrimenti false
        indici=np.argwhere(v==True).flatten()     #Costruisci un array 2D contenente gli indici degli elementi di v posti a 1. Con flatten si definisce l'array in 1 dimensione
        #Split dell'array appena costruito in k pezzi
        new_index=np.array_split(indici,k)
        for j in range(len(new_index)):
            indiciYTrain[j]=np.append(indiciYTrain[j],new_index[j])
    for i in range(k):      #shuffle dei dati
        indiciYTrain[i]=np.random.permutation(indiciYTrain[i])
    return indiciYTrain

#Funzione che definisce la fase di forward_propagation, cioè fase di calcolo dei valori principali della rete:
#restituisce o l'insieme degli output di tutti gli strati e le derivate delle funzioni d'attivazione di tutti gli strati
#o l'unico output della rete
def forward_propagation(network,X,X1,flag):
    Pesi=get_pesi_rete(network)
    Bias=get_bias_rete(network)
    Funzioni_Attivazione=get_funzione_attivazione_rete(network)
    profondita=network['Profondità']
    if flag==0:     #forward_propagation per il semplice calcolo di y della rete (l'output)
        z=X
        for i in range(profondita):
            #per ogni livello calocla l'input tramite prodotto scalare tra W e Z + il baias
            a=np.matmul(Pesi[i],z)+Bias[i]
            #Calcolo dell'output tramite applicazione della funzione d'attivazione sull'input appena calcolato
            z=Funzioni_Attivazione[i](a,0)
        return z
    else:
        a=[]
        z=[]
        derivate_correnti=[]
        z.append(X1)
        #Ciclo che calcola l'input (tramite prodotto scalare e bias esplicito) e l'output tramite l'applicazione della funzione d'attivazione
        for i in range(profondita):
            #Calcolo dell'input di uno strato e salvataggio una struttura dedicata
            a.append(np.matmul(Pesi[i],z[i])+Bias[i])
            #Calcolo dell'output: salvatggio in due strutture dedicate, una per gli output e un'altra per il caloclo delle derivate
            #Questo perchè ci occorrerà in fase di back_propagation la derivata della funzione d'attivazione
            output_corrente, derivata_output_corrente = Funzioni_Attivazione[i](a[i],1)
            derivate_correnti.append(derivata_output_corrente)
            z.append(output_corrente)
    return z,derivate_correnti

#Funzione che definisce la fase di back_propagation: calcolo dei delta e derivate della funzione d'errore
def back_propagation(network, t, error_function,output,derivate):
    Pesi=network['Pesi']
    Bias= network['Bias']
    profondita=network['Profondità']
    #Calcolo della derivata della funzione d'errore sull'ultimo valore, cioè l'output
    derivata_k=error_function(output[-1],t,1)
    #Calcolo del delta valore riferito allo strato d'output
    delta_valori=[]
    delta_valori.insert(0,derivate[-1]*derivata_k)
    #Partendo dall'ultimo strato e salendo, calcolo gli altri delta value
    for i in range(profondita-1,0,-1):
        #Si segue la seguente formula (per il calcolo di DeltaH): sommatoria(Wkh*Deltak)
        delta_corrente=derivate[i-1]*np.matmul(Pesi[i].transpose(),delta_valori[0])
        delta_valori.insert(0,delta_corrente)
    #Calcolo delle derivate tramite legge locale: Derivata di E(sull'n-esima coppia rispetto ai parametri)=Delta(i)*Z(j)
    derivate=[]
    bias_derivate=[]
    for i in range(0,profondita):
        derivata_corrente=np.matmul(delta_valori[i],output[i].transpose())
        derivate.append(derivata_corrente)
        bias_derivate.append(np.sum(delta_valori[i],1,keepdims=True))
    return derivate,bias_derivate

#Calcolo dell'accuratezza: numero casi giusti/numero casi totali
def accuratezza(y,t):
    numero_casi_totali=t.shape[1]
    z=error_function.softMax(y)
    #Il caso è detto giusto se il target rilasciato dalla rete = al target vero
    return np.sum(z.argmax(0)==t.argmax(0))/numero_casi_totali

def accuratezza_rete(network,X,t):
    y=forward_propagation(network,X,X,0)
    return accuratezza(y,t)

#Funzione che restituisce i pesi della rete
def get_pesi_rete(network):
    W=network['Pesi']
    return W

#Funzione che restituisce i bias della rete
def get_bias_rete(network):
    Bias=network['Bias']
    return Bias

#Funzione che restituisce le funzioni d'attivazione applicati ai vari strati della rete
def get_funzione_attivazione_rete(network):
    Funzione_Attivazione=network['Funzione d attivazione']
    return Funzione_Attivazione