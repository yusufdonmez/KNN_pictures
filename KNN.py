# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 15:14:10 2018

@author: computer
"""

import numpy as np
from utils import load_CIFAR10
from utils import visualize_CIFAR

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y
    
    print("Eğitilecek veri(resim) sayısı:",X.shape[0])
    
  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    
    #Buradan ise test veri sayısını belirliyoruz...
    test_veri_sayisi=9900
    num_test = X.shape[0]
    print("Test veri sayısı:", num_test-test_veri_sayisi)
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test-test_veri_sayisi):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred
print("Veriler okunuyor...")
Xtr, Ytr, Xte, Yte = load_CIFAR10('./data') # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
print("Veri okunması bitti. \nSistem eğitimi başlıyor...")

nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
print("Sistem eğitimi bitti.")

print("Eğitilmiş fotoğraf örnekleri:")
visualize_CIFAR(Xtr,Ytr,10)

print("Sistem testi başladı...")
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)

print("Test edilmiş veri(fofoğraf) örnekleri:")
visualize_CIFAR(Xte,Yte_predict,10)

print('Uygulama testi bitti, Sonuç, Doğruluk:%f' % ( np.mean(Yte_predict == Yte)*100 ),)
#print('accuracy: %f' % ( np.mean(Yte_predict == Yte) ))