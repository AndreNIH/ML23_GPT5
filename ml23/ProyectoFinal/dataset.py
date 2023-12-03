import torch
import torchvision
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from pathlib import Path
from utils import to_numpy, to_torch, add_img_text, get_transforms
import cv2
import pathlib

class EyeDataset(Dataset):
    def __init__(self, root_dir, split = "train"):
        self.root_dir = root_dir
        self.split = split
        self.transform = get_transforms(self.split)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir)
        image = read_image(img_path)
        image_name = image_path.split('/')[-1]
        label = image_name[16]
        if self.transform:
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(os.listdir(self.root_dir))


#TODO: Que esto apunte al path verdadero
file_path = pathlib.Path(__file__).parent.absolute()


def get_loader(split, batch_size, shuffle=True, num_workers=0):
    '''
    Get train and validation loaders
    args:
        - batch_size (int): batch size
        - split (str): split to load (train, test or val)
    '''
    dataset = EyeDataset(root=file_path,
                      split=split)
    dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
        )
    return dataset, dataloader


def main():
    # Visualizar de una en una imagen
    split = "train"
    dataset, dataloader = get_loader(split=split, batch_size=1, shuffle=False)
    print(f"Loading {split} set with {len(dataloader)} samples")
    for datapoint in dataloader:
        transformed = datapoint['transformed']
        original = datapoint['original']
        label = datapoint['label']
        emotion = datapoint['emotion'][0]

        # Si se aplico alguna normalizacion, deshacerla para visualizacion
        if dataset.unnormalize is not None:
            # Espera un tensor
            transformed = dataset.unnormalize(transformed)

        # Transformar a numpy
        original = to_numpy(original)  # 0 - 255
        transformed = to_numpy(transformed)  # 0 - 1
        # transformed = (transformed * 255).astype('uint8')  # 0 - 255

        # Aumentar el tamaño de la imagen para visualizarla mejor
        viz_size = (200, 200)
        original = cv2.resize(original, viz_size)
        transformed = cv2.resize(transformed, viz_size)

        # Concatenar las imagenes, tienen que ser del mismo tipo
        original = original.astype('float32') / 255
        np_img = np.concatenate((original,
                                 transformed), axis=1)

        np_img = add_img_text(np_img, emotion)

        #cv2.imshow("img", np_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


if __name__ == '__main__':
    main()