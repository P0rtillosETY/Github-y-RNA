# Carga de los datos del dataset MNIST
import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Configuración de mi red: 784 entradas, 30 neuronas ocultas y 10 de salida
net = network.Network([784, 30, 10])

# Entrenamiento con 30 épocas y tasa de aprendizaje de 3.0
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)