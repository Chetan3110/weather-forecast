import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load data
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Drop rows with missing values
    train.dropna(inplace=True)
    test.dropna(inplace=True)

    return train, test

# Encoding categorical variables
def encode_categorical(train, test):
    le_summary = LabelEncoder()
    le_precip = LabelEncoder()

    # Fit LabelEncoder on combined data to avoid unseen labels
    le_summary.fit(pd.concat([train['Summary'], test['Summary']], axis=0))
    le_precip.fit(pd.concat([train['Precip Type'], test['Precip Type']], axis=0))

    # Encode 'Summary' and 'Precip Type' in both train and test
    train['Summary'] = le_summary.transform(train['Summary'])
    test['Summary'] = le_summary.transform(test['Summary'])

    train['Precip Type'] = le_precip.transform(train['Precip Type'])
    test['Precip Type'] = le_precip.transform(test['Precip Type'])

    return train, test, le_summary, le_precip

# Scale features
def scale_features(train, test):
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(train)
    X_test_scaled = scaler.transform(test)

    return X_train_scaled, X_test_scaled, scaler
