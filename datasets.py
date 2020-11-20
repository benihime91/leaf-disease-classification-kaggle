import torch
import random
import cv2

# set seeds
seed = 42
random.seed(42)
torch.manual_seed(42)

class PlantDataset(torch.utils.data.Dataset):
	def __init__(self, dataframe, transformations):
		self.df = dataframe
		self.transforms = transformations
		
	def __len__(self):
		return len(self.df)
	
	def __getitem__(self, idx):
		image_id = self.df.filePath[idx]
		
		# Read an image with OpenCV
		img = cv2.imread(image_id)
		
		# By default OpenCV uses BGR color space for color images,
		# so we need to convert the image to RGB color space.
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
		# apply transformations to the image
		img = self.transforms(image=img)["image"]
		
		target = self.df.label[idx]
		target = torch.tensor(target)
		
		return img, target
