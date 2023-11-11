import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping
from ucimlrepo import fetch_ucirepo

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fetch dataset
heart_disease = fetch_ucirepo(id=45)

# Data (as pandas dataframes)
X = heart_disease.data.features
Y = heart_disease.data.targets

# Convert to NumPy arrays
X = np.array(X)
Y = np.array(Y)

# Handle missing values (replace NaN with the mean)
X = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).values

# Reshape cholesterol_data to have a single column
cholesterol_data = X[:, X.shape[1] - 1].reshape(-1, 1)

# Ensure Y has the same number of samples as cholesterol_data
Y = Y[:cholesterol_data.shape[0]]

# Coerce the target to be 0 or 1
Y = np.clip(Y, 0, 1)  # This ensures that Y is between 0 and 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cholesterol_data, Y, test_size=0.2, random_state=42)

# Normalize input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Define the neural network architecture with dropout
model = Sequential()
model.add(Dense(128, input_dim=cholesterol_data.shape[1], activation='relu'))
model.add(Dropout(0.5))  # Add dropout layer with a dropout rate of 0.5
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary cross-entropy loss and accuracy, and AUC-ROC metric
model.compile(optimizer=Adam(learning_rate=0.05), loss='binary_crossentropy', metrics=['accuracy', AUC(name='auc')])

# Train the model and monitor validation loss for early stopping
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
loss, accuracy, auc = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}, Test AUC: {auc:.4f}")

# Make predictions on the test set
y_pred = model.predict(X_test)

# If you want binary predictions (0 or 1)
y_pred_binary = (y_pred > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Blues')
plt.show()
