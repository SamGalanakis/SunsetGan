from modelUtils import Generator
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import imageio
from random import random
import cv2 as cv
import librosa
import matplotlib.pyplot as plt


#parameters
fps = 60
hop_length=512
x , sr = librosa.load("GiantSteps.mp3")



duration = librosa.get_duration(x,sr)


seconds = int(duration)
# hop_length= 60

onset_env = librosa.onset.onset_strength(x, sr=sr, hop_length=hop_length, n_fft=2048)

frames = range(len(onset_env))
t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
plt.plot(t, onset_env)
plt.xlim(0, t.max())
plt.ylim(0)
plt.xlabel('Time (sec)')
plt.title('Novelty Function')
plt.show()


imgSize=64
channelsImg=3
channelsNoise=256
numEpochs=200
featuresDiscrim=32
featuresGen=32

myTransforms = transforms.Compose([transforms.ToPILImage(),transforms.Resize((480,480))])

def invNormalize(input,mean,std):

    return input*std + mean

device = torch.device("cuda")

netGenerator = Generator(channelsNoise, channelsImg,featuresGen).to(device)
netGenerator.load_state_dict(torch.load("wandb\last64Sunset\Epoch_195_generatorStateDict.tar"))










total_frames = seconds * fps

name = "video"
write_to = f'output/{name}.mp4' # have a folder of output where output files could be stored.

writer = imageio.get_writer(write_to, format='mp4', mode='I', fps=fps)

tempogram = tempogram[0:channelsNoise]
for second in range(seconds):
    noise = tempogram

with torch.no_grad():

    noiseEnd = torch.randn(channelsNoise,1,1)

    noiseStart= torch.randn(channelsNoise,1,1)
    noiseTotal  = torch.zeros(total_frames,channelsNoise,1,1)
    for index,weight in enumerate(np.linspace(0,1,total_frames)):
        noise = torch.lerp(noiseEnd,noiseStart,weight)
        noiseTotal[index]=noise
    output = netGenerator(noiseTotal.to(device))
    output = invNormalize(output,0.5,0.5).cpu()
    for x in tqdm(range(0,total_frames)):
        sample = myTransforms(output[x]).convert('RGB') 
   
        sample  = cv.GaussianBlur(np.array(sample)[:, :, ::-1],(5,5),0)
        sample = cv.bilateralFilter(sample,9,75,75)
        
        writer.append_data(np.array(sample))




   

    
     
    writer.close()