import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense


tf.keras.backend.set_floatx('float64')
keras.backend.set_floatx('float64')

###### Henon Map
def HenonMap(X,Y,mu,Win,Wout,bin,eta):
    with tf.GradientTape() as tape:
        y_mean_tf = tf.constant(0., dtype = tf.float64)
        y_diam_tf = tf.constant(np.pi, dtype = tf.float64)
        tape.watch(Y)
        Ylast = (Y - y_mean_tf) / y_diam_tf
        if mu is not None:
            Y_V = tf.concat([Ylast, mu], axis = 1)
        else :
            Y_V = Ylast
        hidden = tf.math.tanh(tf.linalg.matmul(Y_V, Win) + bin)
        V = tf.linalg.matmul(hidden,Wout)
    Xout= Y+eta
    Yout=-X+tape.gradient(V,Y)
    return Xout, Yout

'''Define a Henon layer'''
class HenonLayer(layers.Layer): 
    def __init__(self,ni,dim = 3):
        super(HenonLayer, self).__init__()
        init = tf.initializers.GlorotNormal()
        init_zero = tf.zeros_initializer()
        self.dim = dim
        # Defer weight creation to `build` so Keras reliably registers
        # them when the layer is first built. Store shape params.
        self._ni = ni
        self._init = tf.initializers.GlorotNormal()
        # placeholders for weights (will be created in build)
        self.Win = None
        self.Wout = None
        self.bin = None
        self.eta = None

    def build(self, input_shape):
        # Create weights on first build using stored shapes.
        if self.Win is None:
            ni = self._ni
            dim = self.dim
            init = self._init
            self.Win = self.add_weight(name="Win",
                                       shape=[dim, ni],
                                       initializer=init,
                                       dtype=tf.float64,
                                       trainable=True)
            self.Wout = self.add_weight(name="Wout",
                                        shape=[ni, 1],
                                        initializer=init,
                                        dtype=tf.float64,
                                        trainable=True)
            self.bin = self.add_weight(name="bin",
                                       shape=[dim, ni],
                                       initializer=init,
                                       dtype=tf.float64,
                                       trainable=True)
            self.eta = self.add_weight(name="eta",
                                       shape=[dim, 1],
                                       initializer=init,
                                       dtype=tf.float64,
                                       trainable=True)
        super(HenonLayer, self).build(input_shape)
    def call(self,z):
        d = self.dim
        if z.shape[1] != 2*d:
            mu = z[:,2*d:]
        else:
            mu = None
    
        xnext,ynext=HenonMap(z[:,0:d],z[:,d:2*d],mu,self.Win,self.Wout,self.bin,self.eta)
        xnext,ynext=HenonMap(xnext,ynext,mu,self.Win,self.Wout,self.bin,self.eta) 
        xnext,ynext=HenonMap(xnext,ynext,mu,self.Win,self.Wout,self.bin,self.eta) 
        xnext,ynext=HenonMap(xnext,ynext,mu,self.Win,self.Wout,self.bin,self.eta) 
        return tf.concat([xnext,ynext], axis =1)
    
'''Define a HenonNet'''
class HenonNet(Model):
    def __init__(self,unit_list,dim = 3):#
        super(HenonNet, self).__init__()
        self.N = len(unit_list)
        # Keep a Python list for easy iteration, but ALSO assign each
        # sublayer as an attribute on `self` so Keras will register it
        # as a tracked sublayer (and expose its variables).
        self.hlayers = []
        for i in range(self.N):
            ni = unit_list[i]
            hl = HenonLayer(ni, dim)
            # append to list for iteration
            self.hlayers.append(hl)
            # register as attribute so Keras sees it as a sublayer
            setattr(self, f"hl_{i}", hl)
    def call(self, r):
        rout = r
        for i in range(self.N):
            rout = self.hlayers[i](rout)
        return rout 

############################## Data Loading and Preprocessing
def main():
    data = np.load("xsuite_dataset.npz")

    X_raw = data["X"]  # shape (N, 6)
    Y = data["Y"]  # shape (N, 6)

    print("Original X shape:", X_raw.shape)
    print("Original Y shape:", Y.shape)


    Z = X_raw[:, :6] #(x,y, zeta, px, py, delta )
    MU = X_raw[:, 6:9] # (mu_i)i=1:m

    print("MU shape:", MU.shape)

    into_train = int(0.5 * X_raw.shape[0])

    dimensions_to_train = [1] #1 = x,px 2 = y,py 3 = zeta, delta 

    x_train = Z[:into_train, 0:1] # only use x as input, not y, and not the parameters
    px_train = Z[:into_train, 3:4] # only use px as output, not py, and not the parameters
    data = tf.concat([x_train, px_train], axis = 1)

    x_label = Y[:into_train, 0:1] # only use x as output, not y, and not the parameters
    px_label = Y[:into_train, 3:4] # only use px as output

    labels = tf.concat([x_label, px_label], axis = 1)

    '''Convert to tensors
    Z_train = tf.convert_to_tensor(Z[:into_train], dtype=tf.float64)
    MU_train = tf.convert_to_tensor(MU[:into_train], dtype=tf.float64)
    Y_train = tf.convert_to_tensor(Y[:into_train], dtype=tf.float64)
    Z_val = tf.convert_to_tensor(Z[into_train:], dtype=tf.float64)
    MU_val = tf.convert_to_tensor(MU[into_train:], dtype=tf.float64)
    Y_val = tf.convert_to_tensor(Y[into_train:], dtype=tf.float64)
    '''
    print("Training set shapes:", data.shape, labels.shape)
    print("Test set shapes:",data.shape, labels.shape)





    ####### Train the HenonNet
    def schedulerHenon(epoch): 
        if epoch < 100:
            return 1e-1 
        elif epoch < 150:
            return 6e-2 
        elif epoch < 200:
            return 2e-2 
        elif epoch < 300:
            return 5e-3 
        elif epoch < 1000:
            return 1e-3 
        elif epoch < 3000:
            return 4e-4 
        else:
            return 1e-4
    ymean_tf = tf.constant(0., dtype = tf.float64)
    ydiam_tf = tf.constant(np.pi, dtype = tf.float64)
    loss_fun = tf.keras.losses.MeanSquaredError()
    test_model = HenonNet([5,5,5], dim = 1)
    # quick check: show trainable variables before fitting
    print('trainable variables count BEFORE compile:', len(test_model.trainable_variables))
    for v in test_model.trainable_variables[:10]:
        print(v.name, v.shape)
    test_model.compile(optimizer = keras.optimizers.Adam(learning_rate = 1e-4),loss = loss_fun)
    callback = tf.keras.callbacks.LearningRateScheduler(schedulerHenon)
    h = test_model.fit(data, labels, batch_size = 1000, epochs = 5000, verbose=0,callbacks=[callback])

     # Get training loss histories
    training_loss = h.history['loss']
    # Create count of the number of epochs 
    epoch_count = range(1, len(training_loss) + 1) # Visualize loss history
    fig0, ax0 = plt.subplots() 
    plt.plot(epoch_count, training_loss, 'r--') 
    plt.legend(['Training Loss']) 
    ax0.set_title('Loss histroy') 
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')


if __name__ == '__main__':
    main()
