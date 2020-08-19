import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
# from torch.utils.tensorboard import SummaryWriter# eh
from modelUtils import discriminator , generator
import os
import wandb
wandb.init(project="sunset_gan")
from skimage import io
import numpy as np
from PIL import Image
#hyperparameters

lr=0.0002
batchSize=64
imgSize=64
channelsImg=3
channelsNoise=256
numEpochs=100
featuresDiscrim=32
featuresGen=32


myTransforms=transforms.Compose([
    transforms.Resize((imgSize,imgSize)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

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
            transforms.ToPILImage()(sample).show()
        
        else:
            sample.show()
      

# dataset=datasets.ImageFolder(root=r"data",transform=myTransforms)

dataset = imageDataset(r"data\inputData",transform=myTransforms)
dataLoader = DataLoader(dataset,batch_size=batchSize,shuffle=True)
device = torch.device("cuda")

            
print(f"Device : {device}")

netDiscriminator = discriminator(channelsImg,featuresDiscrim).to(device)
netGenerator = generator(channelsNoise, channelsImg,featuresGen).to(device)




optimizerDiscrim = optim.Adam(netDiscriminator.parameters(),lr=lr,betas=(0.5,0.999))
optimizerGenerator = optim.Adam(netGenerator.parameters(),lr=lr,betas=(0.5,0.999))


netDiscriminator.train()
netGenerator.train()

criterion = nn.BCELoss()

realLabel= 1
fakeLabel=0

fixedNoise = torch.randn(64,channelsNoise,1,1).to(device)

wandb.watch(netDiscriminator,criterion=criterion)
wandb.watch(netGenerator,criterion=criterion)

for epoch in range(numEpochs):
    if epoch % 5 ==0 and epoch!=0:
        netDiscriminator.save_state_dict(os.path.join(wandb.run.dir,'discriminatorStateDict.tar'))
        netGenerator.save_state_dict(os.path.join(wandb.run.dir,'generatorStateDict.tar'))
    for batchIndex, data in enumerate(dataLoader):
        #train discrim
        data=data.to(device)
        batchSize =data.shape[0]
        netDiscriminator.zero_grad()
        label= (torch.ones(batchSize)*0.9).to(device) #make unconfident, hack
        output =netDiscriminator(data).reshape(-1)
        lossDiscrimReal=criterion(output,label)
        meanConfidence = output.mean().item()   #for logging
        
        noise= torch.randn(batchSize,channelsNoise,1,1).to(device)
        fake=netGenerator(noise)
        label = (torch.ones(batchSize)*0.1).to(device) #hack as before should be 0

        
        output= netDiscriminator(fake.detach()).reshape(-1)
        lossDiscrimFake = criterion(output,label)
        
        lossDiscrim = lossDiscrimFake +lossDiscrimReal
        lossDiscrim.backward()
        optimizerDiscrim.step()
        #train generator

        netGenerator.zero_grad()
        label= torch.ones(batchSize).to(device)
        output = netDiscriminator(fake).reshape(-1)
        lossGen = criterion(output,label)
        lossGen.backward()
        optimizerGenerator.step()



        if batchIndex % 100 ==0:
            print(f"Epoch [{epoch}/{numEpochs}] Batch {batchIndex}/{len(dataLoader)}, Loss D: {lossDiscrim:.4f} | Loss G: {lossGen:.4f}")
            wandb.log({"epoch": epoch,"batchIndex":batchIndex,"lossDiscrim":lossDiscrim,"lossGen":lossGen," meanConfidence": meanConfidence })

            with torch.no_grad():
                fake = netGenerator(fixedNoise)
                imgGridReal= torchvision.utils.make_grid(data[:32],normalize=True)
                imgGridFake = torchvision.utils.make_grid(fake[:32],normalize=True)
      
                wandb.log({"gridReal": wandb.Image(transforms.ToPILImage()(imgGridReal.cpu()), caption="gridReal")})
                wandb.log({"gridFake": wandb.Image(transforms.ToPILImage()(imgGridFake.cpu()), caption="gridFake")})