import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import pickle

model = tf.keras.models.load_model('sitma.h5')

# Test verisi klasörünü belirleyin
test_dir = 'test'

# Uninfected ve Infected görüntülerini al
uninfected_images = []
parasitized_images = []

for category in os.listdir(test_dir):
    category_dir = os.path.join(test_dir, category)
    for img in os.listdir(category_dir):
        img_path = os.path.join(category_dir, img)
        img_array = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img_array)
        if category == 'Uninfected':
            uninfected_images.append(img_array)
        else:
            parasitized_images.append(img_array)

# Normalizasyon
uninfected_images = np.array(uninfected_images) / 255.0
parasitized_images = np.array(parasitized_images) / 255.0

selected_uninfected_images = uninfected_images[np.random.choice(len(uninfected_images), 3)]
selected_parasitized_images = parasitized_images[np.random.choice(len(parasitized_images), 3)]

uninfected_preds = model.predict(selected_uninfected_images)
parasitized_images_preds = model.predict(selected_parasitized_images)
# Görüntüleri göstermek
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.subplot(3, 2, 2 * i + 1)
    plt.imshow(selected_uninfected_images[i])
    plt.title(f"Uninfected Resim, Tahmin: {'Uninfected' if uninfected_preds[i] > 0.5 else 'Parasitized'}")
    print(uninfected_preds[i])
    plt.xticks([]), plt.yticks([])

    plt.subplot(3, 2, 2 * i + 2)
    plt.imshow(selected_parasitized_images[i])
    plt.title(f"Parasitized Resim, Tahmin: {'Parasitized' if parasitized_images_preds[i] < 0.5 else 'Uninfected'}")
    print(parasitized_images_preds[i])
    plt.xticks([]), plt.yticks([])

plt.show()
# Kaydedilen history verisini yükleyin
with open('history.pkl', 'rb') as f:
    history = pickle.load(f)

# Eğitim geçmişi grafiğini çizme fonksiyonu
def plot_history(history):
    plt.figure(figsize=(12, 6))
    
    # Doğruluk grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.title('Eğitim ve Doğrulama Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()

    # Kayıp grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Eğitim Kaybı')
    plt.plot(history['val_loss'], label='Doğrulama Kaybı')
    plt.title('Eğitim ve Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Grafiklerin yazdırılması
plot_history(history)



