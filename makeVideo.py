


from PIL import Image
from tqdm import tqdm
import numpy as np
import imageio
from random import random
import os





imgDir = r"wandb\last64Mnist\media\images"

imgPaths=[os.path.join(imgDir,f) for f in os.listdir(imgDir) if os.path.isfile(os.path.join(imgDir, f))]
imgPaths = [x for x  in imgPaths if "Fake" in x]
imgPaths = sorted(imgPaths,key = lambda x: int(x.split("_")[1]))
nImages = len(imgPaths)



name = "trainingVideoMnist"
write_to = f'output/{name}.mp4' # have a folder of output where output files could be stored.
fps=2
writer = imageio.get_writer(write_to, format='mp4', mode='I', fps=fps)

for imagePath in tqdm(imgPaths):
    writer.append_data(np.array(Image.open(imagePath)))
writer.close()
