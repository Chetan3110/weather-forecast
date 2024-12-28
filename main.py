from src.data_preprocessing import load_data, encode_categorical, scale_features
from src.feature_engineering import add_cyclic_features, create_sequences
from src.model_building import build_gru_model, tune_hyperparameters
from src.model_training import train_model
from src.evaluate_model import evaluate_model
from src.utils import check_gpu

# Paths to datasets
train_path = 'data/train.csv'
test_path = 'data/test.csv'

# Step 1: Load and preprocess data
train, test = load_data(train_path, test_path)

# Add cyclic features
train['Formatted Date'] = pd.to_datetime(train['Formatted Date'], errors='coerce')
test['Formatted Date'] = pd.to_datetime(test['Formatted Date'], errors='coerce')

train = add_cyclic_features(train, 'Formatted Date')
test = add_cyclic_features(test, 'Formatted Date')

# Encode categorical features
train, test, _, _ = encode_categorical(train, test)

# Scale features
X_train = train.drop(columns=['Formatted Date', 'Daily Summary'])
y_train = train['Daily Summary']
X_test = test.drop(columns=['Formatted Date', 'Daily Summary'])
y_test = test['Daily Summary']

X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

# Create sequences
X_train_seq, y_train_seq = create_sequences(X_train_scaled, sequence_length=7)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, sequence_length=7)

# Step 2: Tune hyperparameters
check_gpu()
tuner, best_hps = tune_hyperparameters(X_train_seq, y_train_seq)

# Step 3: Build and train the best model
best_model = tuner.hypermodel.build(best_hps)
trained_model, _ = train_model(best_model, X_train_seq, y_train_seq)

# Step 4: Evaluate the model
evaluate_model(trained_model, X_test_seq, y_test_seq)
