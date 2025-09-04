# =========================================================
# Protein Sequence Classification using CNN (AlzParkNet)
# =========================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping, ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv('final_seq_data.csv')

# -------------------------------
# 2. Preprocessing
# -------------------------------

# Define amino acid vocabulary and mapping
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_int = {aa: idx+1 for idx, aa in enumerate(amino_acids)}  # Reserve 0 for padding

# Encode protein sequences into integer arrays
def encode_sequence(seq, maxlen=200):
    return [aa_to_int.get(aa, 0) for aa in seq[:maxlen]] + [0]*(maxlen - len(seq[:maxlen]))

# Apply encoding to dataset
df['encoded'] = df['Sequence'].apply(lambda x: encode_sequence(str(x)))

# Prepare features (X) and labels (y)
X = np.array(df['encoded'].tolist())
y = df['Label'].astype(int).values.reshape(-1, 1)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Save encoder for later use (inference)
joblib.dump(encoder, 'label_encoder.joblib')

# -------------------------------
# 3. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -------------------------------
# 4. CNN Model Architecture
# -------------------------------
model = models.Sequential([
    layers.Embedding(input_dim=len(amino_acids)+2, output_dim=64, input_length=200),

    layers.Conv1D(64, 7, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.2),

    layers.Conv1D(256, 5, activation='relu'),
    layers.MaxPooling1D(2),

    layers.Conv1D(128, 5, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dropout(0.3),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),

    layers.Dense(4, activation='softmax')  # 4 classes
])

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks (early stopping + best model checkpoint)
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, min_delta=0.001, restore_best_weights=True)

# -------------------------------
# 5. Handle class imbalance
# -------------------------------
class_weight_values = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['Label']),
    y=df['Label']
)
class_weights = dict(zip(np.unique(df['Label']), class_weight_values))

# -------------------------------
# 6. Train the model
# -------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    verbose=1,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint]
)

# Save final trained model
model.save('alzparknet_cnn.h5')
model.summary()

# -------------------------------
# 7. Inference (Prediction Demo)
# -------------------------------
# Load model and encoder for prediction
model = tf.keras.models.load_model('alzparknet_cnn.h5')
encoder = joblib.load('label_encoder.joblib')

# Sequence encoding function reused
def encode_sequence(seq, maxlen=200):
    return [aa_to_int.get(aa, 0) for aa in seq[:maxlen]] + [0]*(maxlen - len(seq[:maxlen]))

# Example input from user
sequence = input("Enter protein sequence: ").upper()
encoded = np.array([encode_sequence(sequence)])

# Prediction
prediction = model.predict(encoded)
predicted_class = encoder.inverse_transform(prediction)[0][0]

# Label mapping
label_map = {
    1: "Parkinson's Disease",
    2: "Alzheimer's Disease",
    3: "Other Neurodegenerative",
    4: "Healthy"
}
print("Prediction:", label_map[predicted_class])

# -------------------------------
# 8. Model Evaluation
# -------------------------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
class_names = ["Parkinson's", "Alzheimer's", "Other", "Healthy"]

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Training vs Validation Accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

# Training vs Validation Loss
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

# F1 Scores per Class
f1_scores = f1_score(y_true, y_pred_classes, average=None)
plt.figure(figsize=(6, 4))
sns.barplot(x=[0, 1, 2, 3], y=f1_scores)
plt.ylim(0, 1)
plt.title("F1 Scores per Class")
plt.ylabel("F1 Score")
plt.show()

# Classification Report
print("Classification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Confidence vs Prediction Accuracy
confidences = np.max(y_pred, axis=1)
correct = (y_pred_classes == y_true)

plt.figure(figsize=(8, 6))
plt.scatter(confidences[correct], [1]*np.sum(correct), color='green', label='Correct', alpha=0.6)
plt.scatter(confidences[~correct], [0]*np.sum(~correct), color='red', label='Wrong', alpha=0.6)
plt.yticks([0, 1], ['Wrong', 'Correct'])
plt.xlabel("Model Confidence")
plt.title("Confidence vs Prediction Accuracy")
plt.legend()
plt.grid(True)
plt.show()