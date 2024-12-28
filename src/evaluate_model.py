def evaluate_model(model, X_test, y_test):
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {test_mae}")
    return test_loss, test_mae
