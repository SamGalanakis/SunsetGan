from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from skimage import io
import torchvision
class imageDataset(Dataset):
       

    def __init__(self, root_dir, transform=None):
        """
        Args:
 
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
    
        self.root_dir = root_dir
        self.transform = transform
        self.imgPaths= [os.path.join(self.root_dir,f) for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        self.imgPaths=[image for image in self.imgPaths if Image.open(image).mode == "RGB"]
      

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, idx):
  
        sample = Image.open(self.imgPaths[idx])
        if self.transform:
            sample = self.transform(sample)

        return sample
    def showSample(self,idx,transformed=True):
        sample = Image.open(self.imgPaths[idx])
        if transformed:
            sample = self.transform(sample)
            torchvision.transforms.ToPILImage()(sample).show()
        
        else:
            sample.show()