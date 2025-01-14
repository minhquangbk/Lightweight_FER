import torch
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def opencv_loader(path):
    img = cv2.imread(path)
    return img

class DataTransform(object):
    def __init__(self, mean, std, target_size):
        self.transform = {
            'train': transforms.Compose([
                    transforms.RandomResizedCrop(target_size, scale=(0.8, 1.2)),
                    transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
            ]),
            'val': transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
            'test': transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        }
    def __call__(self, img, phase='train'):
        return self.transform[phase](img)
    
class MyDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, phase='train', loader=opencv_loader):
        print('root: ', root)
        super(MyDataset, self).__init__(root)
        self.transform = transform
        self.phase = phase
        self.loader = loader
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        # print('path: ', path)
        try:
            img = self.loader(path) 
            img = Image.fromarray(img)
            img_transformed = self.transform(img, self.phase)
            return img_transformed, target
        except Exception as e:
            print('path ', path)
            print('target ', target)
            print("errror: ", e)

if __name__ == "__main__":
    train_root_path = '/Data/Hoang/emotions/dataset/FERG_v1/test'
    mean, std = 0, 255
    target_size = 256
    transform = DataTransform(mean, std, target_size)
    train_set = MyDataset(train_root_path, transform, phase='train')
    img, label = train_set[0][0], train_set[0][1]
    img = img.numpy()
    cv2.imshow('a', img[0])
    # for i, im in enumerate(img):
    #     # print(img.shape)
    #     np_img = im[0].numpy()
    #     print(np_img.shape)
    #     cv2.imshow(f'n{i}', np_img)
    cv2.waitKey(0)
    
    # # Run only one time
    # split_dataset()