import PIL
import os
from matplotlib import pyplot as plt

def main():
    root_dir = os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, "chest_xray"))
    train_dir = os.path.join(root_dir, "train")

    height_sizes = []
    width_sizes = []
    proportions = []

    for root, _, files in os.walk(train_dir):
        for f in files:
            img_path = os.path.join(root,f)
            image = PIL.Image.open(img_path)

            width, height = image.size
            height_sizes.append(height)
            width_sizes.append(width)
            proportions.append(width/height)

    shape = (350, 250)

    fig = plt.figure(figsize=(15,5))
    fig.suptitle(f"Odlucene dimenzije: {shape}", weight= "bold")
    rows = 1
    cols = 3

    height = {
        "array" : height_sizes, 
        "min" : min(height_sizes),
        "avg" : sum(height_sizes)/len(height_sizes)
    }

    width = {
        "array" : width_sizes, 
        "min" : min(width_sizes),
        "avg" : sum(width_sizes)/len(width_sizes)
    }

    prop = {
        "array" : proportions,
        "avg" : sum(proportions)/len(proportions)
    }

    fig.add_subplot(rows,cols, 1)
    plt.hist(height["array"])
    plt.title("Visina")
    plt.text(1900, 1675, f"Min: {height['min']}")
    plt.text(1900, 1625, f"Avg: {height['avg']:.0f}")

    fig.add_subplot(rows,cols, 2)
    plt.hist(width["array"])
    plt.title("Sirina")
    plt.text(1900, 1425, f"Min: {width['min']}")
    plt.text(1900, 1375, f"Avg: {width['avg']:.0f}")

    fig.add_subplot(rows,cols, 3)
    plt.hist(prop["array"])
    plt.title("Proporcija (Sirina : Visina)")
    plt.text(2.6, 2000, f"Avg: {prop['avg']:.1f}")
    plt.show()

if __name__ == "__main__":
    main()