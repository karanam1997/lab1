import math
import csv
import numpy as np

hlayer_neurons_no = []
w_file_name = []
b_file_name = []
crossdata_val = []
i_weights_val = []
i_bias_val =[]
def file_read(h):
    f = open(h, 'r')
    temp = csv.reader(f)
    return list(temp)


class Neuron:

    def __init__(self,bias):
        self.bias = bias
        self.weights = []

    def out(self, inputs):
        total = 0
        for i in range(len(inputs)):
           total += (self.weights[i]*inputs[i])
        total += self.bias
        self.a_out = 1/ (1 + np.exp(-total))
        print(total, self.a_out)
        return self.a_out
    
    def error(self, t_out):
        return 0.5 *(t_out - self.a_out)
    
    def delta(self, t_out):
        return ((-(t_out - self.a_out))*(self.a_out*(1 - self.a_out)))

class NeuronL:

    def __init__(self, n_num,bias):
        self.bias = bias
        self.neurons = []
        for i in range(n_num):
            self.neurons.append(Neuron(self.bias[i]))
            
    def forward_cycle(self, inputs):
        l_out = np.zeros(len(self.neurons))
        for n in self.neurons:
            l_out.append(n.out(inputs))
        return l_out        

    
class NeuralNet:
    Learning_rate = 0.7
    Momentum = 0.3

    

    def __init__(self, inputs_no, h_layer_no, o_layer_no, hlayer_weights = None, hlayer_bias = None, olayer_weights = None, olayer_bias = None):

        self.inputs = inputs_no
        
        self.hlayer_bias = np.array(hlayer_bias, dtype = np.float)
        self.olayer_bias = np.array(olayer_bias, dtype = np.float)
        self.h_layer = NeuronL(h_layer_no, self.hlayer_bias)
        self.o_layer = NeuronL(o_layer_no, olayer_bias)

        hw_delta = [0]* len(hlayer_weights)
        hold_weights = [0]* len(hlayer_weights)
        ow_delta = [0]* len(olayer_weights)
        oold_weights = [0]* len(olayer_weights)

        print(hlayer_weights)
        
        for h in range(h_layer_no):
                self.h_layer.neurons[h].weights.append(hlayer_weights[h,:])
                
        for o in range(o_layer_no):                          
                self.o_layer.neurons[o].weights.append(olayer_weights[o,:])

   

    def forward(self, inputs):
        h_outs = self.h_layer.forward_cycle(inputs)
        return self.o_layer.forward_cycle(h_outs)

    def train(self, t_inputs, a_outputs):
        self.forward(t_inputs)

        for i in range(len(self.o_layer.neurons)):
            o_delta[i] = self.o_layer.neurons.delta(self.a_outputs[i])

        for h in range(len(self.h_layer.neurons)):
            for o in range(len(sef.o_layer.neurons)):
                temp += o_delta[o]*self.o_layer.neurons[o].weights
            h_delta[h] = (temp * (self.h_layer.neurons.a_out * ( 1 - self.h_layer.neurons.a_out)))

        for o in range(len(self.o_layer.neurons)):
            for w_o in range(len(self.olayer_weights)):
                temp = o_delta[i] * self.o_layer.neurons.inputs(w_o)
                oold_weight[w_o] = self.o_layer.neurons[o].weights[w_o]
                self.o_layer.neurons[o].weights[w_o] += (self.Learning_rate*temp) + (Momentum * w_delta[w_o])                
                ow_delta[w_o] = self.o_layer.neurons[o].weights[w_o] - old_weights[w_o]
                
        for h in range(len(self.h_layer.neurons)):
            for w_h in range(len(self.hlayer_weights)):
                temp = h_delta[i] * self.h_layer.neurons.inputs(w_h)
                hold_weight[w_h] = self.h_layer.neurons[o].weights[w_h]
                self.o_layer.neurons[o].weights[w_h] += (self.Learning_rate*temp) + (Momentum * w_delta[w_h])                
                hw_delta[w_h] = self.o_layer.neurons[o].weights[w_h] - old_weights[w_h]
        
    def o_print():
        print("############# W1 (updated) #############")
        for h in range(len(self.h_layer.neurons)):
            print(self.h_layer.neurons.weights)

        print("############# b1 (updated) #############")
        for h in range(len(self.h_layer.neurons)):
            print(self.h_layer.neurons.bias)

        print("############# W2 (updated) #############")
        for h in range(len(self.o_layer.neurons)):
            print(self.o_layer.neurons.weights)

        print("############# b1 (updated) #############")
        for h in range(len(self.h_layer.neurons)):
            print(self.o_layer.neurons.bias)
        
                         
    
print("##################################\nBackpropagation of MLP\nAuthor: Phanindra Karanam\nLAB1A\n##################################")
inputs_no = int(input("Enter no.of inputs for the Neural Network: "))
outputs_no = int(input("Enter no.of outputs for the Neural Network: "))
hlayer_no = int(input("Enter no.of nodes in hidden layer for Neural Network: "))
               
crossdata_file_name = str(input("Enter the name of the Cross data file:"))
crossdata_val = np.array(file_read(crossdata_file_name), dtype = np.float)
input_val = np.array(crossdata_val[:,:3], dtype = np.float)
dout_val = np.array(crossdata_val[:,3:],dtype = np.float)


        
for i in range(2):
    print("Enter name of the weights file #",i,end = " ")
    h = str(input()) 
    w_file_name.append(h)
    
i_to_h_weights_val = np.array(file_read(w_file_name[0]),dtype = np.float)
h_to_o_weights_val = np.array(file_read(w_file_name[1]),dtype = np.float)

for i in range(2):
    print("Enter name of the bias file #",i,end = " ")
    h = str(input()) 
    b_file_name.append(h)

i_to_h_bias_val = np.array(file_read(b_file_name[0]),dtype = np.float)
h_to_o_bias_val = np.array(file_read(b_file_name[1]),dtype = np.float)

Neural = NeuralNet(inputs_no,hlayer_no,outputs_no,i_to_h_weights_val,i_to_h_bias_val,h_to_o_weights_val,h_to_o_bias_val)
for e in range(len(input_val)):
    Neural.train(input_val[e],dout_val[e])

    
              
