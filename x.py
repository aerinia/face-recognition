import os
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import random
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Specify the path to the extracted dataset folder
dataset_folder = "lfw"

# Step 1: Load the LFW dataset
lfw_people = fetch_lfw_people(data_home=dataset_folder, min_faces_per_person=70, resize=0.4)
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=82)

# Step 3: Perform dimensionality reduction using PCA
n_components = 150
pca = PCA(n_components=n_components, whiten=True, svd_solver='randomized').fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Step 4: Train a SVM classifier
print("Fitting the classifier to the training set")
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = SVC(kernel='rbf', class_weight='balanced')
clf = clf.fit(X_train_pca, y_train)

# Step 5: Predict and evaluate the model
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))


# Step 6: Display some test images and their predicted labels
def plot_gallery(images, titles, h, w, n_row=3, n_col=4, title_text="Gallery"):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.suptitle(title_text, size=16)
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

    for i in range(n_row * n_col):
        if i >= len(images):
            break
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# Define the title function
def title(y_pred, y_true, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_true[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


# Separate correct and incorrect predictions
correct_indices = [i for i in range(len(y_test)) if y_pred[i] == y_test[i]]
incorrect_indices = [i for i in range(len(y_test)) if y_pred[i] != y_test[i]]

# Randomly select 12 indices from each category for display
num_display = 12
random_correct_indices = random.sample(correct_indices, min(num_display, len(correct_indices)))
random_incorrect_indices = random.sample(incorrect_indices, min(num_display, len(incorrect_indices)))

# Prepare images and titles for correct predictions
correct_images = X_test[random_correct_indices]
correct_titles = [title(y_pred, y_test, target_names, i) for i in random_correct_indices]

# Prepare images and titles for incorrect predictions
incorrect_images = X_test[random_incorrect_indices]
incorrect_titles = [title(y_pred, y_test, target_names, i) for i in random_incorrect_indices]

# Plot the gallery with correct predictions
plot_gallery(correct_images, correct_titles, h=50, w=37, title_text="Correct Predictions")
plt.show()

# Plot the gallery with incorrect predictions
plot_gallery(incorrect_images, incorrect_titles, h=50, w=37, title_text="Incorrect Predictions")
plt.show()