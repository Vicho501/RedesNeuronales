import numpy as np
from keras.datasets import mnist
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score


class Layer:
    """Base class for a layer."""
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input_data):
        """Computes the output of a layer for a given input."""
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        """Computes dE/dX for a given dE/dY (and updates parameters if any)."""
        raise NotImplementedError


class FCLayer(Layer):
    """Fully connected layer."""
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class ActivationLayer(Layer):
    """Activation layer."""
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

class ConvLayer(Layer):
    """Convolutional layer."""
    def __init__(self, num_filters, kernel_size, stride, padding, activation=None):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = np.random.rand(num_filters, kernel_size, kernel_size) - 0.5
        self.bias = np.random.rand(num_filters) - 0.5
        self.activation = activation

    def forward_propagation(self, input_data):
        # Implement convolutional forward pass
        self.input=input_data
        if self.activation:
            self.output = self.activation(self.input)
        else:
            self.output = self.input  
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        # Implement convolutional backward pass
        pass

class PoolingLayer(Layer):
    """Base class for pooling layers."""
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward_propagation(self, input_data):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError


class MaxPoolingLayer(PoolingLayer):
    """Max pooling layer."""
    def __init__(self, pool_size, stride):
        super().__init__(pool_size, stride)

    def forward_propagation(self, input_data):
        self.input = input_data
        output = np.zeros((input_data.shape[0], input_data.shape[1] // self.stride, input_data.shape[2] // self.stride))
        for i in range(0, input_data.shape[1], self.stride):
            for j in range(0, input_data.shape[2], self.stride):
                output[:, i // self.stride, j // self.stride] = np.max(input_data[:, i:i + self.pool_size, j:j + self.pool_size], axis=(1, 2))
        self.output = output
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.zeros(self.input.shape)
        for i in range(0, self.input.shape[1], self.stride):
            for j in range(0, self.input.shape[2], self.stride):
                max_idx = np.argmax(self.input[:, i:i + self.pool_size, j:j + self.pool_size], axis=(1, 2))
                input_error[:, i:i + self.pool_size, j:j + self.pool_size] = output_error[:, i // self.stride, j // self.stride][:, np.newaxis, np.newaxis] * (max_idx == np.arange(self.pool_size * self.pool_size)[:, np.newaxis, np.newaxis])
        return input_error


class AveragePoolingLayer(PoolingLayer):
    """Average pooling layer."""
    def __init__(self, pool_size, stride):
        super().__init__(pool_size, stride)

    def forward_propagation(self, input_data):
        self.input = input_data
        output = np.zeros((input_data.shape[0], input_data.shape[1] // self.stride, input_data.shape[2] // self.stride))
        for i in range(0, input_data.shape[1], self.stride):
            for j in range(0, input_data.shape[2], self.stride):
                output[:, i // self.stride, j // self.stride] = np.mean(input_data[:, i:i + self.pool_size, j:j + self.pool_size], axis=(1, 2))
        self.output = output
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.zeros(self.input.shape)
        for i in range(0, self.input.shape[1], self.stride):
            for j in range(0, self.input.shape[2], self.stride):
                input_error[:, i:i + self.pool_size, j:j + self.pool_size] = output_error[:, i // self.stride, j // self.stride][:, np.newaxis, np.newaxis] / (self.pool_size ** 2)
        return input_error

class FlattenLayer(Layer):
    """Flatten layer."""
    def __init__(self):
        pass

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = input_data.reshape(input_data.shape[0], -1)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = output_error.reshape(self.input.shape)
        return input_error

class DropoutLayer(Layer):
    def __init__(self, rate):
        # 'rate' es la fracción de neuronas que se apagan, p.ej., 0.2 significa que el 20% se apagarán
        super().__init__()
        self.rate = rate
        self.mask = None

    def forward_propagation(self, input_data, training=True):
        if training:
            # Crear una máscara binaria aleatoria de la misma forma que los datos de entrada
            self.mask = np.random.binomial(1, 1 - self.rate, size=input_data.shape)
            # Apagar las neuronas según la máscara y escalar los valores restantes
            self.output = input_data * self.mask / (1 - self.rate)
        else:
            # En la fase de evaluación, no se usa Dropout, simplemente pasamos los datos
            self.output = input_data
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        # La máscara también se aplica en la retropropagación
        return output_error * self.mask / (1 - self.rate)

    def backward_propagation(self, output_error, learning_rate):
        # Implementar la retropropagación
        # Este es un ejemplo simplificado, necesitarás implementarlo adecuadamente
        input_error = np.zeros(self.input.shape)
        for f in range(self.filters):
            for i in range(self.output.shape[1]):
                for j in range(self.output.shape[2]):
                    input_error[:, i:i + self.kernel_size[0], j:j + self.kernel_size[1]] += (
                        output_error[f, i, j] * self.weights[f]
                    )
                    self.weights[f] -= learning_rate * output_error[f, i, j] * self.input[:, i:i + self.kernel_size[0], j:j + self.kernel_size[1]]
                    self.bias[f] -= learning_rate * output_error[f, i, j]
        return input_error

# Clase EarlyStopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.wait = 0

    def should_stop(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience



class Network:
    """Neural network."""
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer: Layer):
        """Adds a layer to the network."""
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        """Sets the loss function and its derivative."""
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        """Predicts the output for a given input."""
        input_data = np.array([[x] for x in input_data])
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result
    def fit(self, x_train, y_train, epochs, learning_rate, x_val=None, y_val=None, early_stopping=None):
        x_train = np.array([[x] for x in x_train])
        samples = len(x_train)
        # Loop de Entrenamiento
        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                err += self.loss(y_train[j], output)
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            # calculamos el error promedio entre nodos de salida.
            err = np.mean(err)
            # Imprimomos el error promedio de cada época, más que nada
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

              # Early stopping check if validation data is provided
            if x_val is not None and y_val is not None and early_stopping is not None:
                val_loss = self.validate(x_val, y_val)  # Assuming you have a validate method
                print(f"Validation loss: {val_loss}")
                if early_stopping.should_stop(val_loss):
                    print("Early stopping triggered!")
                    break
    
    def validate(self, x_val, y_val):
        """Calculates the loss on the validation set."""
        x_val = np.array([[x] for x in x_val])
        val_loss = 0
        for j in range(len(x_val)):
            output = x_val[j]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            val_loss += self.loss(y_val[j], output)
        return np.mean(val_loss) / len(x_val)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def mse(y_true, y_pred):
    """Calculates the Mean Squared Error."""
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    """Calculates the derivative of the Mean Squared Error."""
    return 2 * (y_pred - y_true) / y_true.size




# No necesitamos tantos datos.
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#random.seed(123) # Vamos a controlar la aleatoriedad en adelante. 
X, y = zip(*random.sample(list(zip(X_train, y_train)), 2000))

# Sí necesitamos que la forma de X sea la de un vector, en lugar de una matriz. 
X, y = np.array(X, dtype='float64'), np.array(y, dtype='float64')
X = np.reshape(X, (X.shape[0], -1))

# Normalizamos Min-Max
X= MinMaxScaler().fit_transform(X)

# Dividimos la muestra en dos, una para entrenar y otra para testing, como tenemos 
# muestra de sobra nos damos el lujo de testear con la misma cantidad que entrenamos.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

# Necesitamos que y_train sea un valor categórico, en lugar de un dígito entero.
y_train_value = y_train # Guardaremos y_train como valor para un observación más abajo.
from keras.utils import to_categorical
y_train = to_categorical(y_train)

# Necesitamos identificar cuantos nodos tiene nuestra entrada, y eso depende del tamaño de X.
entrada_dim = len(X_train[0])

# Crear instancia de Network
model = Network()

# Agregamos capas al modelo
model.add(FCLayer(entrada_dim, 16))               
model.add(ActivationLayer(tanh, tanh_prime))
model.add(FCLayer(16, 10))                                                   
model.add(ActivationLayer(sigmoid, sigmoid_prime))

# Usar el modelo creado
model.use(mse, mse_prime)
model.fit(X_train, y_train, epochs=20, learning_rate=0.1)

# Usamos el modelo para predecir sobre el conjunto de prueba
y_hat = model.predict(X_test)

# Transformamos la salida en un vector one-hot encoded, es decir 0s y un 1. 
for i in range(len(y_hat)):
    y_hat[i] = np.argmax(y_hat[i][0])

# Reportamos los resultados del modelo
matriz_conf = confusion_matrix(y_test, y_hat)

# The following lines are causing the issue:
# model.add(ConvLayer(num_filters=32, kernel_size=3, stride=1, padding=1))
# model.add(MaxPoolingLayer(pool_size=2, stride=2))
# model.add(FlattenLayer())
# model.add(FCLayer(input_size=128, output_size=10))
# model.add(DropoutLayer(rate=0.2))

# Instead, if you want to add more fully connected layers:
model.add(FCLayer(10, 10)) # Example of another fully connected layer
model.add(ActivationLayer(sigmoid, sigmoid_prime))

X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=123)  # Adjust test_size as needed
# Inicializamos EarlyStopping
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

model.fit(
    X_train_full,
    y_train_full,
    epochs=30,
    learning_rate=0.1,
    x_val=X_val,
    y_val=y_val,
    early_stopping=early_stopping,
)

print('MATRIZ DE CONFUSIÓN para modelo ANN')
print(matriz_conf,'\n')
print('La exactitud de testeo del modelo ANN es: {:.3f}'.format(accuracy_score(y_test,y_hat)))