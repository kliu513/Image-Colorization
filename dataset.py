from torchvision import transforms
from torchvision import transforms, datasets, models
from PIL import Image
from skimage import color
import numpy as np


root = "./Dataset/"
# our version of torch dataset for places 365
class Places365(datasets.VisionDataset):
    # Parameters:
    # resize: bool, whether to resize or not.
    # train: bool, whether it is a training set or not
    # split: double, what is the train-test split
    # resample: int, used in PIL.Image.resize to indicate which resample algo
    # mode: str(can only be 'lab', 'rgb'), to indicate which representation to represent our image
    def __init__(self, resize_shape, root=root, resize = True, train = True,
                 split = 0.98, resample = 3, mode = 'lab'):
        super(Places365, self).__init__(root)
        self.root = root
        self.split = split
        self.train = train
        self.resize = resize
        self.HW = resize_shape
        self.resample = resample
        self.mode = mode
        self.file_path_list = open(root + "places365_train_standard.txt").readlines()
            
    def __getitem__(self, index):
        if not self.train:
            index += int(self.split*len(self.file_path_list))
        file_path, label = self.file_path_list[index].split(' ')
        if self.resize:
            path = root + '/data_256' + file_path
            img = Image.open(path).resize(self.HW, resample = self.resample)
        if self.mode == 'lab':
            lab = color.rgb2lab(img)
            return np.transpose(lab[:,:,1:], (2,0,1)), lab[None,:,:,0], int(label[:-1])
        if self.mode == 'rgb':
            return np.array(img).transpose((2,0,1)), np.array(img.convert("L")[None,:,:]), int(label[:-1])
    
    def __len__(self):
        return int(len(self.file_path_list) * self.split)
   
   
# we only use the validation set of ILSVRC2011, which has 50,000 images, and has name like "ILSVRC2011_val_00000001.jpg"
def get_index(total_len = 8, total_num = 50000):
    results = []
    for i in range(1, total_num+1):
        str_num = str(i)
        idx = "ILSVRC2011_val_" + "0"*(total_len - len(str_num)) + str_num + ".JPEG"
        results.append(idx)
    return results

class ImageNet(datasets.VisionDataset):
    def __init__(self, resize_shape, root=root, resize = True,
                 resample = 3, mode = 'rgb'):
        super(ImageNet, self).__init__(root)
        self.root = root
        self.resize = resize
        self.HW = resize_shape
        self.resample = resample
        self.mode = mode
        self.idx = get_index()
        self.file_path_list = open(root + "ILSVRC2011_validation_ground_truth.txt").readlines()
            
    def __getitem__(self, index):
        img = Image.open(root + "ImageNet/val/" + self.idx )
        if np.array(img).shape < (self.resize_shape,self.resize_shape,3):
            self.__getitem__(index+1)
        if self.resize:
            img = img.resize(self.HW, resample = self.resample)
        if self.mode == 'lab':
            lab = color.rgb2lab(img)
            return np.transpose(lab[:,:,1:], (2,0,1)), lab[None,:,:,0], int(label[:-1])
        if self.mode == 'rgb':
            return np.array(img).transpose((2,0,1)), np.array(img.convert("L")[None,:,:]), int(self.file_path_list[:-1])
    
    def __len__(self):
        return len(self.file_path_list)
    
