import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# # Veri setinin ana klasörü
# data_dir = 'sitma/cell_images'
# parasitized_dir = os.path.join(data_dir, "Parasitized")
# uninfected_dir = os.path.join(data_dir, "Uninfected")

# # Yeni eğitim ve test klasörleri
# output_dir = "sitma/split_cell_images"
# train_dir = os.path.join(output_dir, "train")
# test_dir = os.path.join(output_dir, "test")

# # Eğitim ve test için alt klasörleri oluştur
# for split in [train_dir, test_dir]:
#     os.makedirs(os.path.join(split, "Parasitized"), exist_ok=True)
#     os.makedirs(os.path.join(split, "Uninfected"), exist_ok=True)

# # Görüntü dosyalarını toplama
# data = []
# for img in os.listdir(parasitized_dir):
#     if img.endswith(".png"):  # PNG formatındaki görüntüler
#         data.append((os.path.join(parasitized_dir, img), "Parasitized"))
# for img in os.listdir(uninfected_dir):
#     if img.endswith(".png"):
#         data.append((os.path.join(uninfected_dir, img), "Uninfected"))

# # Veriyi karıştır ve train-test split
# train_files, test_files = train_test_split(data, test_size=0.2, random_state=42)

# # Dosyaları taşıma fonksiyonu
# def copy_files(file_list, dest_dir):
#     for file_path, label in file_list:
#         dest_label_dir = os.path.join(dest_dir, label)
#         shutil.copy(file_path, dest_label_dir)

# # Eğitim ve test dosyalarını taşı
# copy_files(train_files, train_dir)
# copy_files(test_files, test_dir)

datagen = ImageDataGenerator(rescale=1./255)

trainGenerator = datagen.flow_from_directory(
    directory="sitma/split_cell_images/train",
    target_size=(128, 128),
    class_mode='binary',
    batch_size=32
)

test_datagen = ImageDataGenerator(rescale=1./255)

testGenerator = test_datagen.flow_from_directory(
    directory="sitma/split_cell_images/test",
    target_size=(128, 128),
    class_mode='binary',
    batch_size=32
)

# Model tanımlama
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping ekleme
early_stop = EarlyStopping(monitor='val_loss', patience=3)

# Modeli eğitme
history = model.fit(
    trainGenerator,
    epochs=20,
    validation_data=testGenerator,
    callbacks=[early_stop]  # Early stopping callback'i burada ekleniyor
)
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)


model.save("sitmav4.h5")
