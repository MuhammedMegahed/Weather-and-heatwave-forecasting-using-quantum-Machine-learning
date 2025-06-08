import torch
import torch.nn as nn

import pennylane as qml


class QLSTM(nn.Module):
    def __init__(self, #Basically the initialization of the class
                input_size, #The size of the input
                hidden_size, #The size of the hidden state (Number of LSTM Hidden Units)
                n_qubits=4, #The number of qubits
                n_qlayers=1, #The number of entanglement layers
                batch_first=True, #Whether the batch is the first dimension
                return_sequences=False, #Whether to return the sequences
                return_state=False,
                backend="default.qubit"):
        super(QLSTM, self).__init__()
        
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size   #The size of the concatenated input and hidden state
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        #self.dev = qml.device("default.qubit", wires=self.n_qubits)
        #self.dev = qml.device('qiskit.basicaer', wires=self.n_qubits)
        #self.dev = qml.device('qiskit.ibm', wires=self.n_qubits)
        # use 'qiskit.ibmq' instead to run on hardware


        
        """
        The wires for the forget, input, update, and output blocks 
            They create seperate Quantum Circuits for each LSTM Gate
                1. allows for parallel processing
                2. allows for independent quantum computations for each gate
        """
        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]
        
        
        """
        The devices for the forget, input, update, and output blocks 
            They create seperate Quantum Circuits for each LSTM Gate
                1. enables quantum computations on different backends
                2. each device can run its quantum circuit independently
                3. allows for hardware flexibility
        """
        self.dev_forget = qml.device(self.backend, wires=self.wires_forget) # Decides what info to forget from cell state
        self.dev_input = qml.device(self.backend, wires=self.wires_input) # Decides what new info to write into cell state
        self.dev_update = qml.device(self.backend, wires=self.wires_update) # Creates new candidate values to be added to cell state
        self.dev_output = qml.device(self.backend, wires=self.wires_output) # Decides what output to produce
        
        
        """
        this curcuit gates just to defines quantum operations for each gate
        They all follow the same structure:
            1. Encode classical data into qubit states using angle embedding.
            2. Apply trainable quantum layers (entanglement + rotations).
            3. Measure the result using Pauli-Z expectation values.
        """
        def _circuit_forget(inputs, weights):
            qml.templates.BasisEmbedding(inputs, wires=self.wires_forget) # Encodes classical data into qubit states using angle embedding.
            qml.templates.CVNeuralNetLayers(weights, wires=self.wires_forget) # Applies trainable quantum layers (entanglement + rotations).
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget] # Measures the result using Pauli-Z expectation values.
        self.qlayer_forget = qml.QNode(_circuit_forget, self.dev_forget, interface="torch")

        def _circuit_input(inputs, weights):
            qml.templates.BasisEmbedding(inputs, wires=self.wires_input)
            qml.templates.CVNeuralNetLayers(weights, wires=self.wires_input)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_input]
        self.qlayer_input = qml.QNode(_circuit_input, self.dev_input, interface="torch")

        def _circuit_update(inputs, weights):
            qml.templates.BasisEmbedding(inputs, wires=self.wires_update)
            qml.templates.CVNeuralNetLayers(weights, wires=self.wires_update)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]
        self.qlayer_update = qml.QNode(_circuit_update, self.dev_update, interface="torch")

        def _circuit_output(inputs, weights):
            qml.templates.BasisEmbedding(inputs, wires=self.wires_output)
            qml.templates.CVNeuralNetLayers(weights, wires=self.wires_output)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]
        self.qlayer_output = qml.QNode(_circuit_output, self.dev_output, interface="torch")

        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        self.clayer_in = torch.nn.Linear(self.concat_size, n_qubits)
        self.VQC = {
            'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            'input': qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
            'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)
        #self.clayer_out = [torch.nn.Linear(n_qubits, self.hidden_size) for _ in range(4)]

    def forward(self, x, init_states=None):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        '''
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size)  # hidden state (output)
            c_t = torch.zeros(batch_size, self.hidden_size)  # cell state
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]
            
            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # match qubit dimension
            y_t = self.clayer_in(v_t)

            f_t = torch.sigmoid(self.clayer_out(self.VQC['forget'](y_t)))  # forget block
            i_t = torch.sigmoid(self.clayer_out(self.VQC['input'](y_t)))  # input block
            g_t = torch.tanh(self.clayer_out(self.VQC['update'](y_t)))  # update block
            o_t = torch.sigmoid(self.clayer_out(self.VQC['output'](y_t))) # output block

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
            
