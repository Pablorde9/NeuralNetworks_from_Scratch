import numpy as np #linear algebra
import pandas as pd # read file
from matplotlib import pyplot as plt # to plot the results
import os

# --- Starting variables ---
MODEL_FILE = 'model_weights.npz'
DATA_FILE = 'train.csv'


def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"Erro: Arquivo '{DATA_FILE}' não encontrado.")
        print("Por favor, baixe o dataset do Kaggle e coloque na pasta raiz.")
        return None, None, None, None

    print("Carregando dataset...")
    data = pd.read_csv(DATA_FILE)
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)

    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n] / 255.

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n] / 255.

    return X_train, Y_train, X_dev, Y_dev


# --- Neural Networks Functions ---
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(0, Z)


def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T


def deriv_ReLU(Z):
    return Z > 0


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            acc = get_accuracy(get_predictions(A2), Y)
            print(f"Iteration: {i} | Accuracy: {acc:.4f}")
    return W1, b1, W2, b2


# --- Saving Data ---
def save_model(W1, b1, W2, b2):
    np.savez(MODEL_FILE, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"Modelo salvo em {MODEL_FILE}")


def load_model():
    data = np.load(MODEL_FILE)
    return data['W1'], data['b1'], data['W2'], data['b2']


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)


def test_prediction(index, X, Y, W1, b1, W2, b2):
    current_image = X[:, index, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = Y[index]

    print(f"Prediction: {prediction[0]}")
    print(f"Label: {label}")

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


# --- Main ---
if __name__ == "__main__":
    X_train, Y_train, X_dev, Y_dev = load_data()
    if X_train is not None:
        if os.path.exists(MODEL_FILE):
            print("\nCarregando modelo pré-treinado...")
            W1, b1, W2, b2 = load_model()
            print("Modelo carregado com sucesso!")
        else:
            X_train, Y_train, X_dev, Y_dev = load_data()
            print("\nIniciando treinamento do zero...")
            W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 5000)
            save_model(W1, b1, W2, b2)
    # Quick Test
    print("\n--- Teste de Previsão ---")
    test_prediction(0, X_train, Y_train, W1, b1, W2, b2)
    test_prediction(1, X_train, Y_train, W1, b1, W2, b2)
