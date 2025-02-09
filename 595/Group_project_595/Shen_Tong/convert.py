import os
import zipfile
import h5py

import numpy as np
import skimage as ski
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the MNIST dataset (already download)
mnist_train = datasets.MNIST(root='/home/patchouli/AMS/595/Group_project_595/data', train=True, transform=mnist_transform, download=False)
mnist_test = datasets.MNIST(root='/home/patchouli/AMS/595/Group_project_595/data', train=False, transform=mnist_transform, download=False)

train = mnist_train.train_data
test_data = mnist_test.test_data
test_labels = mnist_test.test_labels

# Load BSDS dataset
BSDS_path = os.path.abspath('./data/BSDS500-master.zip')

bg_dataset = []
with zipfile.ZipFile(BSDS_path, 'r') as zip:

    file_names = [f for f in zip.namelist() if f.startswith('BSDS500-master/BSDS500/data/images/train/') and f.endswith('.jpg')]   
    for name in file_names:
        try:
            with zip.open(name) as file:
                bg = ski.io.imread(file)
                bg_dataset.append(bg)

        except Exception as e:
            print(f'Error loading image {name}:{e}')
            continue

print(f'Background dataset has {len(bg_dataset)} images')

def compose_image(mnist, background):

    h, w, _ = background.shape
    mh, mw, _ = mnist.shape
    x = np.random.randint(0, h-mh)
    y = np.random.randint(0, w-mw)

    bg = background[x:x+mh, y:y+mw]

    return np.abs(bg - mnist).astype(np.uint8)

def gen_mnistm(dataset):

    
    output = np.zeros([dataset.shape[0], 28, 28, 3], np.uint8)
    for i in range(dataset.shape[0]):
 
        bg_img = bg_dataset[np.random.randint(len(bg_dataset))]

        d = ski.color.gray2rgb(dataset[i])
        d = compose_image(d, bg_img)
        output[i] = d

    return output
 
mnistm_train = gen_mnistm(train)
mnistm_test_data = gen_mnistm(test_data)

mnistm_dir = os.path.abspath("./data")
h5_path = os.path.join(mnistm_dir, 'mnistm.h5')

with h5py.File(h5_path, 'w') as h5file:
    h5file.create_dataset('train', data=mnistm_train)
    h5file.create_dataset('test_data', data=mnistm_test_data)
    h5file.create_dataset('test_label', data=test_labels)