# Vanilla Network per la classificatione di immagini

<p align= "center">
<img src=https://github.com/kekkokalko/Vanilla-Network-per-la-classificatione-di-immagini/assets/94131849/ad5f3841-5de4-45c0-a5e1-b7190261bfac>

    

## Progetto universitario proposto per l'esame di Machine Learning (Mod B. Neural Network & Deep Learning).

La traccia di tale progetto è la seguente:

PARTE A:
• Progettazione ed implementazione di una libreria di funzioni per:
    ◦ simulare la propagazione in avanti di una rete neurale multi-strato full-connected. Con
      tale libreria deve essere possibile implementare reti con più di uno strato di nodi interni
      e con qualsiasi funzione di attivazione per ciascun strato.
    ◦ la realizzazione della back-propagation per reti neurali multi-strato, per qualunque scelta
      della funzione di attivazione dei nodi della rete e la possibilità di usare almeno la somma
      dei quadrati o la cross-entropy con e senza soft-max come funzione di errore.
      
PARTE B:
Dato il dataset “minist” di immagini di cifre scritte a mano (http://yann.lecun.com/exdb/mnist/),
si consideri come input le immagini raw del dataset mnist. Si ha, allora,
un problema di classificazione a C classi, con C=10. Si estragga opportunamente un dataset
globale di N coppie, e lo si divida opportunamente in training e test set (considerare almeno
10000 elementi per il training set e 2500 per il test set). Si fissi la classica discesa del
gradiente come algoritmo di aggiornamento dei pesi. Si studi l'apprendimento di una rete
neurale (ad esempio epoche necessarie per l’apprendimento, andamento dell’errore su
training e validation set, accuratezza sul test) con uno solo strato di nodi interni al variare
della modalità di apprendimento: online, batch e mini-batch. Si faccia tale studio per
almeno 3 dimensioni diverse (numeri di nodi) per lo strato interno. Scegliere e mantenere
invariati le funzioni di attivazione. Se è necessario, per questioni di tempi computazionali e
spazio in memoria, si possono ridurre le dimensioni delle immagini raw del dataset mnist
(ad esempio utilizzando in matlab la funzione imresize)

Il seguente progetto fa uso della libreria Numpy e Matplotlib.

Anno Accademico 2023/2024

