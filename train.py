import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ✅ Correct dataset path
dataset_path = r"C:\Users\syed kashif\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\11E08EAD45BB565DF33241D7B08620584DD0C85C\transfers\2026-11\pranav\major-project\dataset"

print("Using path:", dataset_path)

# 🔍 Debug (check images)
for root, dirs, files in os.walk(dataset_path):
    imgs = [f for f in files if f.lower().endswith(('.jpg','.jpeg','.png'))]
    print(root, "->", len(imgs), "images")

# ✅ Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# ✅ Load dataset
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=8,
    class_mode='categorical'
)

print("Classes:", train_data.class_indices)
print("Total images:", train_data.samples)

# 🔥 Pretrained MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(128,128,3)
)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)

# Output layer
predictions = Dense(train_data.num_classes, activation='softmax')(x)

# ✅ FIXED TYPO HERE
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# Train
model.fit(train_data, epochs=15)

# Save model
model.save("gi_model.keras")


print("✅ Training Done!")