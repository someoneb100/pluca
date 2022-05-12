import PIL
import os
from matplotlib import pyplot as plt

root_dir = os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, "chest_xray"))
train_dir = os.path.join(root_dir, "train")
test_dir = os.path.join(root_dir, "test")
val_dir = os.path.join(root_dir, "val")

height_sizes = []
width_sizes = []
proportions = []

for root, subdirectories, files in os.walk(train_dir):
    for f in files:
        img_path = os.path.join(root,f)
        image = PIL.Image.open(img_path)

        width, height = image.size
        height_sizes.append(height)
        width_sizes.append(width)
        proportions.append(width/height)

fig = plt.figure(figsize=(10,7))
rows = 1
cols = 3

fig.add_subplot(rows,cols, 1)
plt.hist(height_sizes)
plt.title("Visine")
plt.xlabel("Najmanja je " + str(min(height_sizes)))
plt.ylabel("Prosek je " + str(sum(height_sizes)/len(height_sizes)))

fig.add_subplot(rows,cols, 2)
plt.hist(width_sizes)
plt.title("Sirine")
plt.xlabel("Najmanja je " + str(min(width_sizes)))
plt.ylabel("Prosek je " + str(sum(width_sizes)/len(width_sizes)))

fig.add_subplot(rows,cols, 3)
plt.hist(proportions)
plt.title("Proporcije")
plt.xlabel("Najmanja je " + str(min(proportions)))
plt.ylabel("Prosek je " + str(sum(proportions)/len(proportions)))
plt.show()
