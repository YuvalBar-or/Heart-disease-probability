import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, GlobalAveragePooling1D
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import os
from PIL import Image

# Load the dataset
df = pd.read_csv("cleaned_merged_heart_dataset.csv")

# Train-Test Split
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data for Conv1D
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Learning Rate Scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)


# Function to build, visualize, and train the CNN-inspired model with classification report
def build_and_train_cnn(epochs):
    print(f"\n=== Training CNN Model for {epochs} Epochs ===")

    # Build the CNN model
    cnn_model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = cnn_model.fit(X_train, y_train, epochs=epochs, batch_size=16, validation_split=0.2, verbose=1,
                            callbacks=[lr_scheduler])

    # Predict on the test set
    y_pred_proba = cnn_model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Print classification report and AUC score
    print(f"\nClassification Report for {epochs} Epochs:")
    print(classification_report(y_test, y_pred))
    print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.2f}")

    # Save model visualization
    os.makedirs('./model_visualizations', exist_ok=True)
    model_filename = f'./model_visualizations/cnn_model_{epochs}_epochs.png'
    plot_model(cnn_model, to_file=model_filename, show_shapes=True, show_layer_names=True, dpi=100)
    print(f"Model Architecture saved at: {os.path.abspath(model_filename)}\n")

    # Display the model architecture image
    if os.path.exists(model_filename):
        img = Image.open(model_filename)
        img.show()
    else:
        print(f"File not found: {model_filename}")

    # Plot accuracy and loss
    print(f"Displaying accuracy and loss plots for {epochs} epochs...")
    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
    plt.title(f'Accuracy for {epochs} Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='red')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title(f'Loss for {epochs} Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    return cnn_model


# Train and visualize the CNN model for 15, 30, and 60 epochs
cnn_model_15_epochs = build_and_train_cnn(15)
cnn_model_30_epochs = build_and_train_cnn(30)
cnn_model_60_epochs = build_and_train_cnn(60)

