import mnist_loader
import network

# Cargar datos
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Inicializar red (784 entradas, 30 ocultas, 10 salidas)
net = network.Network([784, 30, 10])

# Entrenar
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)