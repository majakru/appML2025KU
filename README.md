# appML2025KU
Code for the final project in Applied Machine Learning at KU in 2025

Remember to use the same PCA on all the data!

Train an encoder to be good at looking at healthy data
Will see the sick cells are different

First PCA on all data
next AutoEncoder trained on only healthy PCA data

one way:
note the loss from encoding and decoding

anther way:
encode only, and project it with UMAP

Optimizing the PCA: try low number and increase untill we get a good result
TRY: PCA one and two just to see withour autoencoder - does it show anything usefull?
- rank the PCA variables - maybe with feature importance - is it the same order?
- Figure out maybe how many PCA components we should use? maybe after 20 PCA components they are not really important anymore. 
  
Optimizing the autoencoder:


