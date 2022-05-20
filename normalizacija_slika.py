from PIL import Image
from os import path, walk, makedirs, remove
from sys import argv
from shutil import rmtree

def normalize_image(old_image: Image, new_shape: "tuple[int, int]") -> Image:
    prop = old_image.width / old_image.height
    if prop < new_shape[0]/new_shape[1]:
        #slika je malo visa i levo-desno treba dodati crno
        resized_im = old_image.resize((int(new_shape[1]*prop), new_shape[1]))
    else:
        #slika je malo sira i gore-dole treba dodati crno
        resized_im = old_image.resize((new_shape[0], int(new_shape[0]/prop)))
    new_im = Image.new("L", new_shape)
    new_im.paste(resized_im, ((new_shape[0]-resized_im.size[0])//2, (new_shape[1]-resized_im.size[1])//2))
    return new_im

def get_images(directory: str):
    for root, _, files in walk(directory):
        for file in files:
            yield path.relpath(path.join(root,file), start = directory)

def normalize_directory(src_dir: str, dest_dir: str, new_shape: "tuple[int, int]") -> None:
    for img_name in get_images(src_dir):
        src_img = Image.open(path.join(src_dir, img_name))
        dest_img = normalize_image(src_img, new_shape)
        dest_path = path.dirname(path.join(dest_dir, img_name))
        if not path.isdir(dest_path):
            makedirs(dest_path)
        dest_img.save(path.join(dest_dir, img_name))


new_shape = (700, 500)

if __name__ == "__main__":
    if(len(argv) != 3):
        print("Two arguments needed: src_dir and dest_dir!")
        print("Operation aborted...")
        exit(1)

    _, src_dir, dest_dir = argv
    if not path.isdir(src_dir):
        print(f"source_dir ({src_dir}): does not exist!")
        print("Operation aborted...")
        exit(1)

    if path.exists(dest_dir):
        response = input(f"dest_dir ({dest_dir}): already exists! Overwrite? [yes/no] ")
        if(response == "yes"):
            if(path.isdir(dest_dir)):
                rmtree(dest_dir)
            else:
                remove(dest_dir)
        else:
            print("Operation aborted...")
            exit(1)
    
    makedirs(dest_dir)
    print(f"New shape of images: {new_shape}")

    normalize_directory(src_dir, dest_dir, new_shape)