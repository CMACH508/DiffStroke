import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import torch.nn.functional as F
import dlib
import random

class Sketchy_data(Dataset):
    def __init__(self, args, mode='train'):
        self.path = args.data_path
        self.mode = mode
        self.train_list = []
        self.test_list = []
        self.get_list()
        self.train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])
        self.test_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        '''
        return: image shape [C,H,W] 0~1
        '''
        if self.mode == 'train':
            photo_path, sketch_path, caption_path = self.train_list[index]
        elif self.mode == 'test':
            photo_path, sketch_path, caption_path = self.test_list[index]
        photo = Image.open(os.path.join(self.path,'rendered_256x256/256x256/photo/tx_000000000000/',photo_path)).convert('RGB')
        #sketch = Image.open(os.path.join(self.path,'rendered_256x256/256x256/sketch/tx_000000000000/',sketch_path)).convert('RGB')
        
        with open(os.path.join(self.path,'rendered_256x256/256x256/photo/caption/',caption_path), "r") as file:
            line = file.readline()
            caption = line.strip()
        if self.mode=='train':
            diff_photo = self.train_transforms(photo)
            image, deformed_image, deformed_mask = deformation(diff_photo.unsqueeze(0),grid_size=np.random.randint(4,8),
                                                                num_changes=np.random.randint(1,5),
                                                                magnitude=np.random.uniform(0.1,0.3), training=True)
            #mask = deformed_image.unsqueeze(0).repeat(1,3,1,1)
            #deformed_image = deformed_image * mask + (1-mask)*image
        else:
            diff_photo = self.test_transforms(photo)
            image, deformed_image, deformed_mask = deformation(diff_photo.unsqueeze(0),grid_size=np.random.randint(4,8),
                                                                num_changes=np.random.randint(1,5),
                                                                magnitude=np.random.uniform(0.1,0.3), training=False)

        return {'pixel_value':image.squeeze(), 'deformed_value':deformed_image.squeeze(),
                 'caption':caption, 'deformed_mask':deformed_mask.repeat(3,1,1)}

    def get_list(self):
        # [(photo,sketch,caption), (photo,sketch,caption),...]
        test_tmp_list = []
        test_set_info = os.path.join(self.path, 'info-06-04/info/testset.txt')    
        with open(test_set_info, "r") as file:
            line = file.readline()
            while line:
                test_tmp_list.append(line.strip())
                line = file.readline()
        print('Loading data')
        for category in os.listdir(os.path.join(self.path,'rendered_256x256/256x256/sketch/tx_000000000000/')):
            img_list = sorted(os.listdir(os.path.join(self.path,'rendered_256x256/256x256/sketch/tx_000000000000/',category)))
            for name in img_list:
                tmp_data =os.path.join(category,name)
                symbol_index = tmp_data.rindex('-')
                if tmp_data[:symbol_index]+'.jpg' not in test_tmp_list:
                    self.train_list.append((tmp_data[:symbol_index]+'.jpg',tmp_data, tmp_data[:symbol_index]+'.txt'))
                else:
                    self.test_list.append((tmp_data[:symbol_index]+'.jpg',tmp_data, tmp_data[:symbol_index]+'.txt'))
        print(f'number of traning set: {len(self.train_list)}, number of test set: {len(self.test_list)}')

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_list)
        else:
            return len(self.test_list)


def initialize_control_points(grid_size, scale=1.0):
    """Initialize the controlp points"""
    points = np.linspace(0, scale, num=grid_size)
    meshx, meshy = np.meshgrid(points, points)
    control_points = np.stack([meshx, meshy], axis=2)
    return control_points

def modify_control_points(control_points, num_changes=5, magnitude=0.1):
    """Randomly modify the control points"""
    for _ in range(num_changes):
        i, j = np.random.randint(0, control_points.shape[0], size=2)
        direction = np.random.rand(2) * 2 - 1
        control_points[i, j] += direction * magnitude
    return control_points

def apply_deformation(image, control_points, original_control_points):
    channels, height, width = image.shape[1:]
    spline_x = RectBivariateSpline(np.linspace(0, 1, control_points.shape[0]),
                                   np.linspace(0, 1, control_points.shape[1]),
                                   control_points[:, :, 0])
    spline_y = RectBivariateSpline(np.linspace(0, 1, control_points.shape[0]),
                                   np.linspace(0, 1, control_points.shape[1]),
                                   control_points[:, :, 1])

    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    map_x = spline_x.ev(grid_y, grid_x)
    map_y = spline_y.ev(grid_y, grid_x)

    grid = np.stack([map_x, map_y], axis=-1)
    grid = (torch.from_numpy(grid).float() - 0.5) * 2  # Normalize to [-1, 1]
    grid = grid.unsqueeze(0)  # Reformat for grid_sample

    return F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)

def create_deformation_mask(control_points, original_control_points, size):
    """Generate mask"""
    deformation_field = np.linalg.norm(control_points - original_control_points, axis=2)
    deformation_field = (deformation_field > 0)  
    

    spline = RectBivariateSpline(np.linspace(0, 1, deformation_field.shape[0]),
                                 np.linspace(0, 1, deformation_field.shape[1]),
                                 deformation_field.astype(float))
    x = np.linspace(0, 1, size[1])
    y = np.linspace(0, 1, size[0])
    mask = spline(y, x)
    mask = mask > 0.05 
    return torch.from_numpy(mask.astype(float)).unsqueeze(0)

def apply_mask(image, mask):
    return image * mask

def deformation(image,grid_size=4,scale=1.0,num_changes=1,magnitude=0.1,training=False):
    control_points = initialize_control_points(grid_size, scale=scale)
    original_control_points = np.copy(control_points)
    control_points = modify_control_points(control_points, num_changes=num_changes, magnitude=magnitude)
    deformed_image = apply_deformation(image, control_points, original_control_points)
    deformation_mask = create_deformation_mask(control_points, original_control_points, image.shape[2:])
    # image (C,H,W)
    if np.random.rand() > 0.5 and training:
        return deformed_image, image, deformation_mask
    else:
        return image,deformed_image,deformation_mask
    

class Custom_dataset(Dataset):
    def __init__(self, args):
        self.path = args.data_path
        self.test_list = []
        self.get_list()
        self.test_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.mask_transforms = transforms.Compose(
        [
            transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def get_list(self):
        image_names = os.listdir(os.path.join(self.path, 'images'))
        num_images = len(image_names)
        print(f"{num_images} images to be edited.")
        for i in range(num_images):
            self.test_list.append(f'{i}')

    def __getitem__(self, index):
        photo = Image.open(os.path.join(self.path,'images',f'{self.test_list[index]}.png')).convert('RGB')
        sketch = Image.open(os.path.join(self.path,'edges',f'{self.test_list[index]}.png')).convert('RGB')     
        #sketch = Image.open(os.path.join(self.path,'edges',f'{3}.png')).convert('RGB')        
        with open(os.path.join(self.path,'captions',f'{self.test_list[index]}.text'), "r") as file:
            line = file.readline()
            text = line.strip()
        #text = ""
        photo = self.test_transforms(photo)
        sketch = self.test_transforms(sketch)

        require_mask = False #'human_mask', 'mask', 'GMask'
        if require_mask == True:
            mask = Image.open(os.path.join(self.path,'human_mask',f'{self.test_list[index]}.png'))
            mask = self.mask_transforms(mask)[0:1,:,:]
            mask = (mask>0.5).float()
            return {'photos': photo, 'sketches':sketch, 'caption': text, 'mask':mask}
        return {'photos': photo, 'sketches':sketch, 'caption': text}

    def __len__(self):
        return len(self.test_list)
    

class Face_data(Dataset):
    def __init__(self, args, mode='train'):
        self.path = args.data_path
        self.mode = mode
        self.train_list = []
        self.test_list = []
        self.img_list = self.get_list()
        self.train_transforms = transforms.Compose(
        [
            transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(1024) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])
        self.test_transforms = transforms.Compose(
        [
            transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
    
    def __getitem__(self, index):
        img_name = self.img_list[index]

        with open(os.path.join(self.path, 'captions', img_name+'.text')) as f:
            caption = f.readline().strip()

        img = Image.open(os.path.join(self.path, 'CelebA-HQ-img', img_name+'.jpg')).convert('RGB')
        if self.mode == 'train':
            img_tensor = self.train_transforms(img)
        elif self.mode == 'test':
            img_tensor = self.test_transforms(img)

        if self.mode == 'train' and np.random.rand() > 0.75:
            image, deformed_image, mask = deformation(img_tensor.unsqueeze(0),grid_size=np.random.randint(6,10),
                                                                    num_changes=np.random.randint(1,3),
                                                                    magnitude=np.random.uniform(0.05,0.15), training=True)
            image, deformed_tensor,mask = image.squeeze(), deformed_image.squeeze(),mask.squeeze()
        else:
            landmarks = self.get_landmarks(os.path.join(self.path, 'CelebA-HQ-img', img_name+'.jpg'))
            if len(landmarks)<5:
                # avoid bug
                landmarks=[(0,0),(0,1),(0,2),(1,0),(1,1),(2,0)]

            displacement, mask = self.generate_landmark_displacement(landmarks, img_tensor.shape[1:3])
            mask = mask.permute(1,0)
            transformed_img = self.apply_transform(img_tensor, displacement)
            deformed_tensor = transformed_img.permute(0,1,3,2).squeeze()
        if self.mode == 'train' and np.random.rand() > 0.5:
            image = deformed_tensor
            deformed_image = img_tensor
            deformed_mask = mask.unsqueeze(0)
        else:
            image = img_tensor
            deformed_image = deformed_tensor
            deformed_mask = mask.unsqueeze(0)
        return {'pixel_value':deformed_image, 'deformed_value':image,
                 'caption':caption, 'deformed_mask':deformed_mask.repeat(3,1,1)}

        # return {'pixel_value':image, 'deformed_value':deformed_image,
        #          'caption':caption, 'deformed_mask':deformed_mask.repeat(3,1,1)}
    
    def __len__(self):
        return len(self.img_list)

    def get_list(self):
        imgs_list = []
        if self.mode=='train':
            with open(os.path.join(self.path,'train_split.txt'), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    imgs_list.append(line.strip().split('.')[0])

        elif self.mode=='test': 
            with open(os.path.join(self.path,'test_split.txt'), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    imgs_list.append(line.strip().split('.')[0])
        print(f'mode: {self.mode}, samples: {len(imgs_list)}')
        return imgs_list
    
    def generate_landmark_displacement(self, landmarks, img_size, num_landmarks=random.randint(3,5), 
                                   sigma=0.1, intensity=random.randint(8,12), grid_density=random.randint(4,10),
                                   threshold=0.01):
        height, width = img_size
        grid_height, grid_width = height // grid_density, width // grid_density
        displacement = torch.zeros(grid_height, grid_width, 2)

        if len(landmarks) > num_landmarks:
            tmp_landmarks = np.random.choice(len(landmarks), num_landmarks, replace=False)
            selected_landmarks = torch.tensor([landmarks[i] for i in tmp_landmarks], dtype=torch.float32)
        else:
            selected_landmarks = torch.tensor(landmarks, dtype=torch.float32)
    
        selected_landmarks = (selected_landmarks / torch.tensor([width, height])) * 2 - 1

        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, steps=grid_height), torch.linspace(-1, 1, steps=grid_width), indexing='ij')
        grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

        for (x_l, y_l) in selected_landmarks:
            distances = torch.norm(grid_points - torch.tensor([x_l, y_l]), dim=1)
            weight = torch.exp(-distances**2 / (2 * sigma**2))
            weight = weight.view(grid_height, grid_width)
            displacement[..., 0] += weight * (x_l - grid_x)
            displacement[..., 1] += weight * (y_l - grid_y)
    
        mask = torch.any(displacement != 0, dim=2).float()

        displacement = torch.clip(displacement, -1, 1)
        displacement = F.interpolate(displacement.permute(2, 0, 1).unsqueeze(0), size=img_size, mode='bilinear', align_corners=True).squeeze(0).permute(1, 2, 0)

        norm_displacement = torch.norm(displacement, dim=2)
        mask = (norm_displacement > threshold).float()
    
        displacement = torch.clip(displacement, -1, 1)
        displacement = F.interpolate(displacement.permute(2, 0, 1).unsqueeze(0), size=img_size, mode='bilinear', align_corners=True).squeeze(0).permute(1, 2, 0)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=img_size, mode='nearest').squeeze()
        return displacement, mask

    def apply_transform(self, image, displacement):
        grid = displacement.unsqueeze(0) + torch.stack(torch.meshgrid([
            torch.linspace(-1, 1, steps=image.shape[1]),
            torch.linspace(-1, 1, steps=image.shape[2])
        ], indexing='ij'), dim=0).permute(1, 2, 0).unsqueeze(0)
        return F.grid_sample(image.unsqueeze(0), grid, mode='bilinear', padding_mode='border',align_corners=False)

    def get_landmarks(self, image_path):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat') 
        image = dlib.load_rgb_image(image_path)
        detections = detector(image)
        landmarks = []
        for detection in detections:
            shape = predictor(image, detection)
            landmarks = [(p.x, p.y) for p in shape.parts()]
        return landmarks
