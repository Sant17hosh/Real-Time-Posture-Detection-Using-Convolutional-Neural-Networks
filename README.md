# ============================
# Posture Detection Training Script
# ============================

# Step 1: Install dependencies (only needed once if not installed)
# !pip install tensorflow scikit-learn opencv-python Pillow matplotlib numpy pandas openpyxl

import os
import zipfile
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ----------------------------
# Step 2: File paths
# ----------------------------
frames_zip = "frames.zip"         # <-- put your 1000+ frames zip here
labels_file = "updated.xlsx"      # <-- your Excel with labels

# Adjust paths if running in Colab
if "COLAB_GPU" in os.environ:
    frames_zip = "/content/frames.zip"
    labels_file = "/content/updated.xlsx"

# Check files exist
if not os.path.exists(frames_zip):
    raise FileNotFoundError(f"❌ {frames_zip} not found")
if not os.path.exists(labels_file):
    raise FileNotFoundError(f"❌ {labels_file} not found")

print(f"✅ Dataset: {frames_zip}")
print(f"✅ Labels: {labels_file}")

# ----------------------------
# Step 3: Extract frames
# ----------------------------
extract_dir = "extracted_frames"
if os.path.exists(extract_dir):
    import shutil
    shutil.rmtree(extract_dir)
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(frames_zip, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

# Collect images
image_files = glob.glob(os.path.join(extract_dir, "**", "*.*"), recursive=True)
image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
image_files = sorted(image_files)

if not image_files:
    raise ValueError("❌ No images found in frames.zip")
print(f"✅ Found {len(image_files)} images")

# ----------------------------
# Step 4: Load Excel labels
# ----------------------------
df = pd.read_excel(labels_file, engine="openpyxl")
df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]

label_columns = ['upright', 'leaning_forward', 'leaning_backward', 'leaning_left', 'leaning_right']
if not all(col in df.columns for col in label_columns):
    raise ValueError(f"❌ Missing required label columns. Found: {df.columns.tolist()}")

print("✅ Excel columns:", df.columns.tolist())
print("📊 First rows:\n", df.head())

# ----------------------------
# Step 5: Match images with labels
# ----------------------------
X, y = [], []

if "filename" in df.columns:
    print("✅ Matching by 'filename' column")
    label_dict = df.set_index("filename")[label_columns].to_dict(orient="index")
    for img_path in image_files:
        fname = os.path.basename(img_path)
        if fname in label_dict:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            X.append(img)
            y.append(list(label_dict[fname].values()))
else:
    print("⚠️ No filename column, matching by order")
    if len(image_files) != len(df):
        raise ValueError(f"❌ Mismatch: {len(image_files)} images vs {len(df)} rows in Excel")
    for i, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        X.append(img)
        y.append(df.iloc[i][label_columns].values)

X = np.array(X)
y = np.array(y)

print(f"✅ Processed {len(X)} images")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# ----------------------------
# Step 6: Split train/val
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)}, Validation: {len(X_val)}")

# ----------------------------
# Step 7: Build CNN model
# ----------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(5, activation="sigmoid")  # 5 posture classes
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# Step 8: Train model
# ----------------------------
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# ----------------------------
# Step 9: Plot training curves
# ----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss over Epochs")
plt.legend()

plt.show()

# ----------------------------
# Step 10: Save model + history
# ----------------------------
model.save("posture_detection_model.h5")
print("✅ Model saved as posture_detection_model.h5")

pd.DataFrame(history.history).to_csv("training_history.csv", index=False)
print("✅ Training history saved as training_history.csv")
