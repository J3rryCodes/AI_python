import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True) #download


class Network:
    def activation(self,a,dec=False):
        if dec == True :
            return a*(1-a)
        return 1/(1+np.exp(-a))
    
    epochs = 6000
    l0 = 784
    l1 = 400
    l2 = 200
    l3 = 100
    l4 = 10
    weight_ih = 2 * np.random.random((l0,l1)) - 1
    weight_hh1 = 2 * np.random.random((l1,l2)) - 1
    weight_hh2 = 2 * np.random.random((l2,l3)) - 1
    weight_ho = 2 * np.random.random((l3,l4)) - 1
    bias_h1 = 2 * np.random.random((1,l1)) - 1
    bias_h2 = 2 * np.random.random((1,l2)) - 1
    bias_h3 = 2 * np.random.random((1,l3)) - 1
    bias_o = 2 * np.random.random((1,l4)) - 1
    learning_rate = 0.07
    
    def feedforword(self,X):
        lay0 = np.array([X])
        lay1 = self.activation(lay0.dot(self.weight_ih) + self.bias_h1)
        lay2 = self.activation(lay1.dot(self.weight_hh1) + self.bias_h2)
        lay3 = self.activation(lay2.dot(self.weight_hh2) + self.bias_h3)
        lay4 = self.activation(lay3.dot(self.weight_ho) + self.bias_o)
        return lay4

    #training data (feedforword + backprop)
    def train(self,Xs,ys):
        epoch_error = 0
        for i in range(len(Xs)*10):
            r = np.random.randint(len(Xs))
            lay0 = np.array([Xs[r]])
            lay1 = self.activation(lay0.dot(self.weight_ih) + self.bias_h1)
            lay2 = self.activation(lay1.dot(self.weight_hh1) + self.bias_h2)
            lay3 = self.activation(lay2.dot(self.weight_hh2) + self.bias_h3)
            lay4 = self.activation(lay3.dot(self.weight_ho) + self.bias_o)
            #error
            error = ys[r] - lay4
            delta_output = (error * self.activation(lay4,True)) * self.learning_rate

            hidden_error3 = delta_output.dot(self.weight_ho.T)
            delta_hidden3 = hidden_error3 * self.activation(lay3,True) * self.learning_rate

            hidden_error2 = delta_hidden3.dot(self.weight_hh2.T)
            delta_hidden2 = hidden_error2 * self.activation(lay2,True) * self.learning_rate

            hidden_error1 = delta_hidden2.dot(self.weight_hh1.T)
            delta_hidden1 = hidden_error1 * self.activation(lay1,True) * self.learning_rate

            
            self.weight_ho+=lay3.T.dot(delta_output)
            self.weight_hh2+=lay2.T.dot(delta_hidden3)
            self.weight_hh1+=lay1.T.dot(delta_hidden2)
            self.weight_ih+=lay0.T.dot(delta_hidden1)

            self.bias_o += delta_output
            self.bias_h1 += delta_hidden1
            self.bias_h2 += delta_hidden2
            self.bias_h3 += delta_hidden3
            epoch_error += abs(max(error[0]))
            if(i % (len(Xs)/10)) == 0:
                print(f'{100-(epoch_error/len(Xs))*1000} % accuracy ---- {(i/(len(Xs)*10))*100} % completed')
                epoch_error = 0

                
    # loading data from disk
    def load_weights(self):
    	data = np.load("NN_data.npz")
    	self.weight_ih = data['weight_ih']
    	self.weight_hh1 = data['weight_hh1']
    	self.weight_hh2 = data['weight_hh2']
    	self.weight_ho = data['weight_ho']
    	self.bias_h1 = data['bias_h1']
    	self.bias_h2 = data['bias_h2']
    	self.bias_h3 = data['bias_h3']
    	self.bias_o = data['bias_o']
    	print("Pre-traind data loaded.........")


Xs = mnist.train.images
ys = mnist.train.labels

nn = Network()

nn.load_weights() # load pre-traind data

#nn.train(Xs,ys) # train the network (it will take a while)


while 1:
    a = int(input(f'Entre an index (0 - {mnist.test.num_examples}) : '))
    test_images , test_labels = mnist.test.next_batch(1)
    print("Predicted Label : ",np.argmax(nn.feedforword(test_images)))
    print("Actual Label    : ",np.argmax(test_labels))
