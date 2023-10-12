import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import L2


def train(X_train, y_train, X_val, y_val, model, epochs, batch_size, verbose=False, patience=0):
    if patience > 0:
        callbacks = [EarlyStopping(monitor='loss', patience=patience)]
    else:
        callbacks = []

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks,
                        workers=-1,
                        verbose=verbose)
    if patience > 0:
        early_stopping = callbacks[0]
        print("Entrenamiento detenido en", early_stopping.stopped_epoch, "epocas")
    return history


def create_model(hidden_neurons: list, lambda_regu: int, n_classes: int, learning_rate: float):
    output_layer = [Dense(units=n_classes, activation="linear",
                          kernel_regularizer=L2(lambda_regu))]

    hidden_layers = [Dense(units=neurons, activation="relu",
                           kernel_regularizer=L2(lambda_regu))
                     for neurons in hidden_neurons]

    layers = hidden_layers + output_layer

    model = Sequential(layers)

    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    return model


def show_train(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], color='blue', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], color='red', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    # Gráfico de pérdida
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], color='blue', label='Training Loss')
    plt.plot(history.history['val_loss'], color='red', label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


def predict(model, X_eval):
    logits = model.predict(X_eval)
    softmax_output = tf.nn.softmax(logits)
    y_pred = tf.argmax(softmax_output, axis=1).numpy()
    return y_pred
