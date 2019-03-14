#XOR problem
import numpy as np #matrix math

#input vlues
X = np.array([[0,0],
			[1,0],
			[0,1],
			[1,1]])
#utputvalues
y = np.array([[0],
			[1],
			[1],
			[0]])
class nuralnetwork:
	no_epoches = 100000
	learnig_rate = 0.001
	ih_weights = 2 * np.random.random((2,3)) - 1 #weights b/w input layer and hidden layer [2x3]
	ho_weights = 2 * np.random.random((3,1)) - 1 #weights b/w hidden layer and output layer [3x2]
	i_bais = 2 * np.random.random((1,3)) - 1 #bias for input layer [1x3]
	h_bias = 2 * np.random.random((1,1)) - 1 #bias for hidden layer [1x1]

	def sigmoid(self,a):
		return 1/(1+np.exp(-a)) #activation function

	def desigmoid(self,a):
		return a*(1-a) #derivative of activation function

	def feedforword(self,x):
		input_layer = x 
		hidden_layer = self.sigmoid(input_layer.dot(self.ih_weights) + self.i_bais) # hidden layer = input layer x inputTOhidden weights [1x3]
		output_layer = self.sigmoid(hidden_layer.dot(self.ho_weights) + self.h_bias) # output layer = hidden layer x hiddenTOoutput weights [1x1]
		return output_layer

	def train(self,X,y):
		for i in range(self.no_epoches):
			r = i%len(X) 
			#feed forword
			input_layer = np.array([X[r]])
			hidden_layer = self.sigmoid(input_layer.dot(self.ih_weights) + self.i_bais) # hidden layer = input layer x inputTOhidden weights [1x3]
			output_layer = self.sigmoid(hidden_layer.dot(self.ho_weights) + self.h_bias) # output layer = hidden layer x hiddenTOoutput weights [1x1]

			#back propogation
			output_error = y[r] - output_layer #[1x1]
			delta_output = output_error * self.desigmoid(output_layer) #[1x1]

			hidden_error = delta_output.dot(self.ho_weights.T) #[1x3]
			delta_hidden = hidden_error*(self.desigmoid(hidden_layer)) #[1x3]

			self.ho_weights += hidden_layer.T.dot(delta_output) #[3x1]

			self.ih_weights += input_layer.T.dot(delta_hidden) #[2x3]
			self.h_bias += delta_output
			self.i_bais += delta_hidden
		print('Final error'+str(abs(output_error)))

nn = nuralnetwork()
nn.train(X,y)
for x in X:
        if(nn.feedforword(x) > 0.5):
                print(f'[{x[0]}] : [{x[1]}] --> 1')
        else:
                print(f'[{x[0]}] : [{x[1]}] --> 0')
	
