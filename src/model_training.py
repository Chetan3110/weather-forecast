from tensorflow.keras.callbacks import EarlyStopping

# Train the model
def train_model(model, X_train, y_train, validation_split=0.2, batch_size=16, max_epochs=50):
    stop_early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=validation_split,
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[stop_early]
    )

    return model, history
