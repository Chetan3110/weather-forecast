from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

# Define the GRU model
def build_gru_model(hp):
    model = Sequential()

    # Add GRU layer with tunable units
    model.add(GRU(
        units=hp.Int('units', min_value=32, max_value=256, step=32),
        activation='relu',
        input_shape=(None, None)  # Adjust input shape dynamically
    ))

    # Add Dropout for regularization
    model.add(Dropout(hp.Choice('dropout_rate', values=[0.2, 0.3, 0.4])))

    # Dense layer
    model.add(Dense(hp.Int('dense_units', min_value=16, max_value=128, step=16), activation='relu'))

    # Output layer
    model.add(Dense(1))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='mse',
        metrics=['mae']
    )

    return model

# Hyperparameter tuning
def tune_hyperparameters(X_train, y_train, validation_split=0.2, max_epochs=50):
    tuner = kt.Hyperband(
        build_gru_model,
        objective='val_mae',
        max_epochs=max_epochs,
        factor=3,
        directory='hyperband_logs',
        project_name='gru_weather_forecasting'
    )

    # Early stopping
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Perform hyperparameter search
    tuner.search(X_train, y_train, validation_split=validation_split, epochs=max_epochs, callbacks=[stop_early])

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return tuner, best_hps
