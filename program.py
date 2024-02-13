import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib.pyplot as plt

# Read the first CSV file
df = pd.read_csv(r'C:\Users\Rahul S\Desktop\dcn pro\for final year.csv')

# Drop unnecessary columns

# Handle non-numeric values in 'Signal_Strength' and 'Resource_Allocation'
df['Signal_Strength'] = pd.to_numeric(df['Signal_Strength'], errors='coerce')
df['Resource_Allocation'] = pd.to_numeric(df['Resource_Allocation'], errors='coerce')

# Identify columns with non-numeric values
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
print("Columns with non-numeric values in the first dataset:", non_numeric_cols)

# Replace NaN values with mean for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Define features and target for the first dataset
X = df.drop(['Resource_Allocation'], axis=1)
y = df['Resource_Allocation']

# Split the first dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features for the first dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data for 1D CNN for the first dataset
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the CNN model for the first dataset
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))  # Linear activation for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# Predict on the test set
y_pred = model.predict(X_test)

# Check for NaN values in predictions
nan_indices = np.isnan(y_pred)
print(f'Number of NaN values in predictions: {np.sum(nan_indices)}')

# Remove NaN values from predictions and corresponding true values
y_test_no_nan = y_test.to_numpy()[~nan_indices.flatten()]
y_pred_no_nan = y_pred[~nan_indices.flatten()]

# Check if there are samples to calculate MSE
if len(y_test_no_nan) > 0:
    # Calculate mean squared error
    mse = mean_squared_error(y_test_no_nan, y_pred_no_nan)
    print(f'Mean Squared Error: {mse}')

    # Generate a graph of actual vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()

    # Generate a scatter plot of actual vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_no_nan, y_pred_no_nan)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')
    plt.show()
else:
    print('No valid samples for calculating Mean Squared Error.')

# Read the second CSV file
df2 = pd.read_csv(r'C:\Users\Rahul S\Desktop\dcn pro\data1.csv')

# Drop unnecessary columns

# Handle non-numeric values in 'Signal_Strength' and 'Resource_Allocation'
df2['Signal_Strength'] = pd.to_numeric(df2['Signal_Strength'], errors='coerce')
df2['Resource_Allocation'] = pd.to_numeric(df2['Resource_Allocation'], errors='coerce')

# Identify columns with non-numeric values
non_numeric_cols_2 = df2.select_dtypes(exclude=[np.number]).columns
print("Columns with non-numeric values in the second dataset:", non_numeric_cols_2)

# Replace NaN values with mean for numeric columns
numeric_cols_2 = df2.select_dtypes(include=[np.number]).columns
df2[numeric_cols_2] = df2[numeric_cols_2].fillna(df2[numeric_cols_2].mean())

# Define features and target for the second dataset
X2 = df2.dropa(['Resource_Allocation'], axis=1)
y2 = df2['Resource_Allocation']

# Split the second dataset into training and testing sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Standardize the features for the second dataset
scaler2 = StandardScaler()
X_train2 = scaler2.fit_transform(X_train2)
X_test2 = scaler2.transform(X_test2)

# Reshape the data for 1D CNN for the second dataset
X_train2 = X_train2.reshape((X_train2.shape[0], X_train2.shape[1], 1))
X_test2 = X_test2.reshape((X_test2.shape[0], X_test2.shape[1], 1))

# Build a separate CNN model for the second dataset
model2 = Sequential()
model2.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train2.shape[1], 1)))
model2.add(MaxPooling1D(pool_size=2))
model2.add(Flatten())
model2.add(Dense(50, activation='relu'))
model2.add(Dense(1, activation='linear'))  # Linear activation for regression

# Compile the second model
model2.compile(optimizer='adam', loss='mean_squared_error')

# Train the second model
history2 = model2.fit(X_train2, y_train2, epochs=30, batch_size=32, validation_data=(X_test2, y_test2), verbose=0)