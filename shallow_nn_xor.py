import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)


@dataclass
class TrainingHistory:
    mse_batch_train: List[float] = field(default_factory=list)
    mse_batch_full: List[float] = field(default_factory=list)
    mse_layer1_batch: List[float] = field(default_factory=list)
    mse_layer2_batch: List[float] = field(default_factory=list)
    mse_per_epoch_train: List[float] = field(default_factory=list)
    mse_per_epoch_full: List[float] = field(default_factory=list)
    classification_error: List[float] = field(default_factory=list)
    W1_history: List[np.ndarray] = field(default_factory=list)
    W2_history: List[np.ndarray] = field(default_factory=list)
    mse_layer1: List[float] = field(default_factory=list)
    mse_layer2: List[float] = field(default_factory=list)


class TwoLayerNN:
    """
    Prosta sieć dwuwarstwowa (warstwa ukryta + wyjściowa) do zadań klasyfikacji binarnej.
    Używamy sigmoidu i błędu MSE, uczymy się metodą spadku gradientu z:
    - momentum
    - adaptacyjnym współczynnikiem uczenia
    - możliwością uczenia mini-batch.
    """

    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int = 1, random_state: Optional[int] = None):
        rng = np.random.RandomState(random_state)

        # Inicjalizacja wag (małe wartości losowe)
        self.W1 = rng.normal(scale=1.0, size=(n_inputs, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.normal(scale=1.0, size=(n_hidden, n_outputs))
        self.b2 = np.zeros((1, n_outputs))

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Zwraca (z1, h, z2, y_hat)."""
        z1 = X @ self.W1 + self.b1
        h = sigmoid(z1)
        z2 = h @ self.W2 + self.b2
        y_hat = sigmoid(z2)
        return z1, h, z2, y_hat

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(0.5 * (y_true - y_pred) ** 2))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, _, _, y_hat = self.forward(X)
        return y_hat

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_full: Optional[np.ndarray] = None,
        y_full: Optional[np.ndarray] = None,
        epochs: int = 1000,
        learning_rate: float = 0.1,
        batch_size: int = 1,  # 1 = online, N = batch, in-between = mini-batch
        momentum: float = 0.0,
        adaptive_lr: bool = False,
        target_mse: Optional[float] = None,
        verbose: bool = False,
    ) -> TrainingHistory:
        """
        Trening sieci dwuwarstwowej.
        X_full/y_full – dane do liczenia błędu na „całym zbiorze uczącym”.
        Jeśli są None, używamy X_train/y_train.
        """
        if X_full is None:
            X_full = X_train
        if y_full is None:
            y_full = y_train

        n_samples, _ = X_train.shape
        if batch_size <= 0 or batch_size > n_samples:
            batch_size = n_samples

        history = TrainingHistory()

        # Momentum – poprzednie przyrosty wag
        v_W1 = np.zeros_like(self.W1)
        v_b1 = np.zeros_like(self.b1)
        v_W2 = np.zeros_like(self.W2)
        v_b2 = np.zeros_like(self.b2)

        current_lr = learning_rate
        prev_epoch_mse = None

        for epoch in range(epochs):
            # Losowa permutacja próbek
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, n_samples, batch_size):
                stop = start + batch_size
                xb = X_shuffled[start:stop]
                yb = y_shuffled[start:stop]

                # Forward
                z1, h, z2, y_hat = self.forward(xb)

                # Backprop – pochodne MSE względem wag
                dL_dy = (y_hat - yb)  # d(MSE)/d(y_hat)
                dL_dz2 = dL_dy * sigmoid_derivative(z2)
                dL_dW2 = h.T @ dL_dz2 / xb.shape[0]
                dL_db2 = np.mean(dL_dz2, axis=0, keepdims=True)

                dL_dh = dL_dz2 @ self.W2.T
                dL_dz1 = dL_dh * sigmoid_derivative(z1)
                dL_dW1 = xb.T @ dL_dz1 / xb.shape[0]
                dL_db1 = np.mean(dL_dz1, axis=0, keepdims=True)
                history.mse_layer1.append(float(np.mean(dL_dz1 ** 2)))
                history.mse_layer2.append(float(np.mean(dL_dz2 ** 2)))

                # Aktualizacja z momentum
                v_W2 = momentum * v_W2 - current_lr * dL_dW2
                v_b2 = momentum * v_b2 - current_lr * dL_db2
                v_W1 = momentum * v_W1 - current_lr * dL_dW1
                v_b1 = momentum * v_b1 - current_lr * dL_db1

                self.W2 += v_W2
                self.b2 += v_b2
                self.W1 += v_W1
                self.b1 += v_b1

                # --- MATLAB-style logging ---
                # MSE na aktualnym batchu (NIE na epokę)
                mse_batch = self.mse(yb, y_hat)
                history.mse_batch_train.append(mse_batch)

                # MSE na pełnym zbiorze (full dataset) po każdym update — jak MATLAB
                _, _, _, y_full_pred_iter = self.forward(X_full)
                mse_full_iter = self.mse(y_full, y_full_pred_iter)
                history.mse_batch_full.append(mse_full_iter)

                # Logowanie deltas dla poszarpanych wykresów warstw
                history.mse_layer1_batch.append(float(np.mean(dL_dz1 ** 2)))
                history.mse_layer2_batch.append(float(np.mean(dL_dz2 ** 2)))

            # Koniec epoki – liczymy błędy na całych zbiorach
            _, _, _, y_train_hat = self.forward(X_train)
            _, _, _, y_full_hat = self.forward(X_full)
            # Ponowny backprop dla pełnego zbioru (tylko do celów diagnostycznych)
            z1_full, h_full, z2_full, y_full_hat = self.forward(X_full)

            dL_dy_full = (y_full_hat - y_full)
            dL_dz2_full = dL_dy_full * sigmoid_derivative(z2_full)
            dL_dh_full = dL_dz2_full @ self.W2.T
            dL_dz1_full = dL_dh_full * sigmoid_derivative(z1_full)

            # history.mse_layer1.append(float(np.mean(dL_dz1_full ** 2)))
            # history.mse_layer2.append(float(np.mean(dL_dz2_full ** 2)))

            mse_train = self.mse(y_train, y_train_hat)
            mse_full = self.mse(y_full, y_full_hat)

            y_pred_labels = (y_full_hat >= 0.5).astype(int)
            class_error = float(np.mean(y_pred_labels != y_full))

            history.mse_per_epoch_train.append(mse_train)
            history.mse_per_epoch_full.append(mse_full)
            history.classification_error.append(class_error)
            history.W1_history.append(self.W1.copy())
            history.W2_history.append(self.W2.copy())

            if adaptive_lr and prev_epoch_mse is not None:
                if mse_train > prev_epoch_mse:
                    current_lr *= 0.9
                else:
                    current_lr *= 1.1
                current_lr = max(0.01, min(current_lr, 1.0))

            prev_epoch_mse = mse_train

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(
                    f"Epoka {epoch+1}/{epochs} | "
                    f"MSE train={mse_train:.6f} | MSE full={mse_full:.6f} | "
                    f"class error={class_error:.3f} | lr={current_lr:.5f}"
                )

            # Wczesne zakończenie uczenia
            if target_mse is not None and mse_train <= target_mse:
                if True:
                    print(f"Przerywamy uczenie po epoce {epoch+1}, osiągnięto MSE_train <= {target_mse}.")
                break
            if epoch == epochs - 1:
                if True:
                    print(f"Osiągnięto maksymalną liczbę epok: {epochs}.")

        return history


# ====== Funkcje pomocnicze do generowania XOR i rysowania wykresów ======

def make_xor_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Klasyczny problem XOR – 4 punkty w 2D."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=float)
    return X, y

def plot_classification_error(history: TrainingHistory, title_prefix: str = "", comment = "") -> None:
    epochs = np.arange(1, len(history.classification_error) + 1)
    plt.figure()
    plt.plot(epochs, history.classification_error)
    plt.xlabel("Epoka")
    plt.ylabel("Błąd klasyfikacji (odsetek)")
    if comment == "":
        plt.title(f"{title_prefix}Błąd klasyfikacji (próg 0.5)")
    else:
        plt.title(f"{title_prefix}Błąd klasyfikacji (próg 0.5) - {comment}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title_prefix}classification_error_{comment}.png")


def plot_weights(history: TrainingHistory, title_prefix: str = "", comment = "") -> None:
    epochs = np.arange(1, len(history.W1_history) + 1)

    W1_stack = np.stack(history.W1_history, axis=0)
    n_in, n_hidden = W1_stack.shape[1], W1_stack.shape[2]

    plt.figure()
    plt.subplot(2, 1, 1)
    for i in range(n_in):
        for j in range(n_hidden):
            plt.plot(epochs, W1_stack[:, i, j], label=f"W1[{i},{j}]" if (i == 0 and j == 0) else None)
    plt.xlabel("Epoka")
    plt.ylabel("Wartość wagi")
    if comment == "":
        plt.title(f"{title_prefix}Ewolucja wag – warstwa 1")
    else:
        plt.title(f"{title_prefix}Ewolucja wag – warstwa 1 - {comment}")
    if n_in * n_hidden > 0:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()

    W2_stack = np.stack(history.W2_history, axis=0)
    n_hidden, n_out = W2_stack.shape[1], W2_stack.shape[2]

    plt.subplot(2, 1, 2)
    for i in range(n_hidden):
        for j in range(n_out):
            plt.plot(epochs, W2_stack[:, i, j], label=f"W2[{i},{j}]" if (i == 0 and j == 0) else None)
    plt.xlabel("Epoka")
    plt.ylabel("Wartość wagi")
    plt.title(f"{title_prefix}Ewolucja wag – warstwa 2")
    if n_hidden * n_out > 0:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title_prefix}weights_{comment}.png")

def plot_mse_matlab(history: TrainingHistory, title_prefix="", comment = ""):
    iters = np.arange(len(history.mse_batch_train))

    plt.figure(figsize=(12, 7))

    plt.subplot(2, 1, 1)
    plt.plot(iters, history.mse_layer1_batch, 'b', linewidth=1)
    if(comment == ""):
        plt.title(f"{title_prefix}Błąd średniokwadratowy sieci")
    else:
        plt.title(f"{title_prefix}Błąd średniokwadratowy sieci - {comment} ")

    plt.ylabel("MSE – warstwa 1")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(iters, history.mse_batch_train, 'b', label="na przykładach uczących", linewidth=1)
    plt.plot(iters, history.mse_batch_full, 'r', label="na całym ciągu uczącym", linewidth=1)
    plt.ylabel("MSE")
    plt.xlabel("Iteracja")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{title_prefix}mse_{comment}.png")

# ====== Przykład: uczenie XOR z wszystkimi wymaganymi dodatkami ======

def example_xor_training(batch_size: 2, momentum: 0.9, adaptive_lr: True, target_mse: float = 1e-3, chart_drawing: bool = True, comment: str = ""):
    X, y = make_xor_dataset()
    nn = TwoLayerNN(n_inputs=2, n_hidden=4, n_outputs=1, random_state=42)

    X_train, X_full, y_train, y_full = X, X, y, y

    history = nn.train(
        X_train=X_train,
        y_train=y_train,
        X_full=X_full,
        y_full=y_full,
        epochs=5000,
        learning_rate=0.3,
        batch_size=batch_size,  # 1 - online, N - batch
        momentum=momentum,  # 0.0 - switched off
        adaptive_lr=adaptive_lr,
        target_mse=target_mse,
        verbose=False,
    )
    y_pred = nn.predict(X)
    # print("Wejścia XOR:")
    # print(X)
    # print("Oczekiwane wyjścia:", y.ravel())
    print("Wyjścia sieci:", nn.predict_proba(X).ravel())
    # print("Klasy (0/1):", y_pred.ravel())

    if chart_drawing:
        plot_classification_error(history, title_prefix="XOR – ", comment=comment)
        plot_weights(history, title_prefix="XOR – ", comment=comment)
        plot_mse_matlab(history, "XOR – ", comment=comment)

        plt.show()



def example_breast_cancer():
    data = load_breast_cancer()
    X = data.data.astype(float)
    y = data.target.astype(float).reshape(-1, 1)  # 0/1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_inputs = X_train.shape[1]
    nn = TwoLayerNN(n_inputs=n_inputs, n_hidden=16, n_outputs=1, random_state=0)

    history = nn.train(
        X_train=X_train,
        y_train=y_train,
        X_full=X_train,
        y_full=y_train,
        epochs=200,
        learning_rate=0.1,
        batch_size=32, # 1 - online, N - batch
        momentum=0.9, #0.0 - switched off
        adaptive_lr=True,
        target_mse=None,
        verbose=True,
    )

    y_test_pred_proba = nn.predict_proba(X_test)
    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
    test_error = float(np.mean(y_test_pred != y_test))
    test_acc = 1.0 - test_error

    print(f"Dokładność na zbiorze testowym (breast_cancer): {test_acc:.3f}")

    plot_mse_matlab(history, "Breast cancer – ")
    plot_classification_error(history, title_prefix="Breast cancer – ")
    plot_weights(history, title_prefix="Breast cancer – ")
    plt.show()


if __name__ == "__main__":
    example_xor_training(batch_size=1, momentum=0, adaptive_lr=False, target_mse=None, chart_drawing=False, comment="")
    example_xor_training(batch_size=1, momentum=0, adaptive_lr=False, target_mse=1e-3, chart_drawing=False, comment="Early stopping")
    example_xor_training(batch_size=1, momentum=0.9, adaptive_lr=False, target_mse=1e-3, chart_drawing=False, comment="Momentum")
    example_xor_training(batch_size=1, momentum=0.9, adaptive_lr=True, target_mse=1e-3, chart_drawing=False, comment="Momentum, adaptive LR")
    example_xor_training(batch_size=2, momentum=0.9, adaptive_lr=True, target_mse=1e-3, chart_drawing=False, comment="mini-batch, momentum, adaptive LR, target MSE")

    # example_breast_cancer()



# def plot_mse(history: TrainingHistory, title_prefix: str = "") -> None:
#     epochs = np.arange(1, len(history.mse_per_epoch_train) + 1)
#
#     plt.figure()
#     plt.plot(epochs, history.mse_per_epoch_train, label="MSE – przykłady uczące")
#     plt.plot(epochs, history.mse_per_epoch_full, label="MSE – cały zbiór uczący", linestyle="--")
#     plt.xlabel("Epoka")
#     plt.ylabel("MSE")
#     plt.title(f"{title_prefix}Błąd MSE w czasie uczenia")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#
# def plot_internal_mse(history: TrainingHistory):
#     epochs = np.arange(1, len(history.mse_layer1) + 1)
#
#     plt.figure(figsize=(10,6))
#
#     plt.subplot(2, 1, 1)
#     plt.plot(epochs, history.mse_layer1, 'b')
#     plt.title("Błąd średniokwadratowy – warstwa 1")
#     plt.ylabel("MSE – warstwa 1")
#     plt.grid(True)
#
#     plt.subplot(2, 1, 2)
#     plt.plot(epochs, history.mse_layer2, 'b', label="na przykładach uczących")
#     plt.ylabel("MSE – warstwa 2")
#     plt.xlabel("Epoka")
#     plt.grid(True)
#
#     plt.tight_layout()
