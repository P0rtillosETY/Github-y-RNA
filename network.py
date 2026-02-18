"""Implementación de una red neuronal "feedforward" que utiliza el algoritmo 
de Descenso de Gradiente Estocástico (SGD) para el aprendizaje."""

#### Librerías
# Librería estándar
import random

# Librerías de terceros
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """
        El parámetro 'sizes' define la arquitectura de la red (ej. [784, 30, 10]).
        Se inicializan los sesgos y pesos de forma aleatoria siguiendo una 
        distribución Gaussiana de media 0 y varianza 1 para romper la simetría.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Se inicializan los sesgos de forma aletoria para cada capa(excepto la entrada)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Se inicializan los pesos conectando cad aneurona de una capa con la siguiente, con valores aleatorios
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        #Esta función procesa la entrada "a" a través de toda la red.
        # Aplicando la fórmula: activavion = sigmoide(peso * entrada + sesgo) 
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        # SGD es el algoritmo de Descenso de Gradiente Estocástico.
        #Entrena la red dividiendo los datos en mini-lotes para mayor eficiencia.
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            #Mezcla los datos en cada época para que la red no aprenda el orden.
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            # Actualizamos la red basándonos en cada mini-lote.
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                # Evaluamos la presión actual con los datos de prueba.
            if test_data:
                print("Época {} : {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Época {} completa".format(j))

    def update_mini_batch(self, mini_batch, eta):
        # Ajusta los pesos y los sesgos usando la regla de actualización del gradiente.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # Calculo de cuánto debe cambiar cada peso (gradiente) usando backprop.
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #Actualización de los pesos finales usando la tasa de aprendizaje (eta).
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # El corazón del aprendizaje: calcula el error de la red hacia atrás
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Propagación hacia adelante para ver qué predice la red. (feedforward)
        activation = x
        activations = [x] # lista para almacenar todas las activaciones, capa por capa
        zs = [] # lista para almacenar todos los vectores z, capa por capa
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # Cálculo del error en la última capa (backward pass)
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # l = 1 es la última capa de neuronas, l = 2 la penúltima, y así sucesivamente (propagare el error)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Evalúa el rendimiento de la red. Retorna la suma de predicciones 
        exitosas comparando la neurona de mayor activación con la etiqueta real.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        #Devuelve el vector de derivadas parciales para las activaciones de salida.
        return (output_activations-y)

#### Funciones Matemáticas Auxiliares
def sigmoid(z):
    #La función sigmoide que normaliza los valores entre 0 y 1
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    #Derivada de la función sigmoide, para el cálculo de gradientes en backprop 
    return sigmoid(z)*(1-sigmoid(z))