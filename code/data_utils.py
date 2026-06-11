import numpy as np
import cv2
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
import torch
from torch.utils.data import Dataset
try:
    from natsort import natsorted
except ImportError:
    def natsorted(seq, key=None):
        def natural_key(item):
            value = key(item) if key is not None else item
            return [int(part) if part.isdigit() else part.lower()
                    for part in re.split(r'(\d+)', str(value))]
        return sorted(seq, key=natural_key)

def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_points3D_text(path):
    """Read points3D.txt file"""
    points3D = {}
    with open(path, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            data = line.split()
            point_id = int(data[0])
            xyz = np.array([float(x) for x in data[1:4]])
            rgb = np.array([int(x) for x in data[4:7]])
            error = float(data[7])
            points3D[point_id] = {
                'xyz': xyz,
                'rgb': rgb,
                'error': error
            }
    return points3D

def read_images_text(path):
    """Read images.txt file and return images sorted by name"""
    images = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # First collect all images
    for i in range(0, len(lines), 2):
        line = lines[i]
        if line[0] == '#':
            continue
        data = line.split()
        image_id = int(data[0])
        qvec = np.array([float(x) for x in data[1:5]])
        tvec = np.array([float(x) for x in data[5:8]])
        camera_id = int(data[8])
        name = data[9]
        
        R = qvec2rotmat(qvec)
        
        images[image_id] = {
            'R': R,
            't': tvec.reshape(3,1),
            'camera_id': camera_id,
            'name': name
        }
    
    # Sort images by name and create new ordered dictionary
    sorted_images = dict(natsorted(images.items(), key=lambda x: x[1]['name']))
    
    return sorted_images

def read_cameras_text(path):
    """Read cameras.txt file"""
    cameras = {}
    with open(path, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            data = line.split()
            camera_id = int(data[0])
            model = data[1]
            width = int(data[2])
            height = int(data[3])
            params = np.array([float(x) for x in data[4:]])
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras

def get_intrinsic_matrix(camera, downsample_factor=1):
    """Get intrinsic matrix from camera parameters"""
    if camera['model'] == 'PINHOLE':
        fx, fy, cx, cy = camera['params']
        fx, fy, cx, cy = fx / downsample_factor, fy / downsample_factor, cx / downsample_factor, cy / downsample_factor
        K = np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])
        return K
    else:
        raise ValueError(f"Camera model {camera['model']} not supported yet")




def look_at_colmap(eye, target, up):
    """OpenCV/COLMAP convention: camera x right, y down, z forward."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    R = np.stack([x, y, z], axis=0)
    t = -R @ eye
    return R.astype(np.float32), t.reshape(3, 1).astype(np.float32)


def build_synthetic_camera_data(image_paths, downsample_factor=1):
    """Build a circular camera path when COLMAP output is unavailable."""
    from PIL import Image

    with Image.open(image_paths[0]) as img:
        width, height = img.size

    camera_angle_x = 0.6911112070083618
    focal = 0.5 * width / np.tan(0.5 * camera_angle_x)
    K = np.array([
        [focal / downsample_factor, 0.0, 0.5 * width / downsample_factor],
        [0.0, focal / downsample_factor, 0.5 * height / downsample_factor],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    radius = 4.0
    elevation = 1.0
    target = np.zeros(3, dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    camera_data = []
    for i in range(len(image_paths)):
        theta = 2.0 * np.pi * i / len(image_paths)
        eye = np.array([
            radius * np.sin(theta),
            elevation,
            radius * np.cos(theta),
        ], dtype=np.float32)
        R, t = look_at_colmap(eye, target, up)
        camera_data.append({"K": K.copy(), "R": R, "t": t})
    return camera_data


def build_synthetic_points(image_path, camera_data, maximum_pts_num=3000):
    """Back-project foreground pixels from one RGBA image into a coarse colored cloud."""
    from PIL import Image

    rgba = np.asarray(Image.open(image_path).convert("RGBA"))
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    ys, xs = np.where(alpha > 16)
    if len(xs) == 0:
        ys, xs = np.where(np.ones(alpha.shape, dtype=bool))

    if len(xs) > maximum_pts_num:
        indices = np.linspace(0, len(xs) - 1, maximum_pts_num).astype(np.int64)
        xs, ys = xs[indices], ys[indices]

    width = rgba.shape[1]
    camera_angle_x = 0.6911112070083618
    focal = 0.5 * width / np.tan(0.5 * camera_angle_x)
    cx, cy = 0.5 * rgba.shape[1], 0.5 * rgba.shape[0]

    R = camera_data[0]["R"]
    t = camera_data[0]["t"].reshape(3)
    cam_center = -R.T @ t
    depth = np.linalg.norm(cam_center)
    x_cam = (xs.astype(np.float32) - cx) / focal * depth
    y_cam = (ys.astype(np.float32) - cy) / focal * depth
    z_cam = np.full_like(x_cam, depth)
    cam_points = np.stack([x_cam, y_cam, z_cam], axis=-1)
    world_points = (R.T @ (cam_points - t).T).T
    colors = rgb[ys, xs].astype(np.float32)
    return torch.as_tensor(world_points).float(), torch.as_tensor(colors).float()


class ColmapDataset(Dataset):
    def __init__(self, data_path, downsample_factor=8, maximum_pts_num=3000, max_views=None):
        """
        Dataset class for COLMAP data
        """
        sparse_path = os.path.join(data_path, "sparse", "0_text")
        images_dir = os.path.join(data_path, "images")
        required_files = ["cameras.txt", "images.txt", "points3D.txt"]
        missing_files = [
            os.path.join(sparse_path, name)
            for name in required_files
            if not os.path.exists(os.path.join(sparse_path, name))
        ]
        if missing_files:
            print(
                "Warning: COLMAP text reconstruction is missing. "
                "Using a synthetic circular camera path and coarse image-derived point cloud."
            )
            self.downsample_factor = downsample_factor
            self.image_paths = natsorted([
                os.path.join(images_dir, name)
                for name in os.listdir(images_dir)
                if name.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            if max_views is not None:
                self.image_paths = self.image_paths[:max_views]
            if not self.image_paths:
                raise FileNotFoundError(f"No images found in {images_dir}")
            self.camera_data = build_synthetic_camera_data(self.image_paths, downsample_factor)
            self.points3D_xyz, self.points3D_rgb = build_synthetic_points(
                self.image_paths[0], self.camera_data, maximum_pts_num
            )
            return

        self.downsample_factor = downsample_factor
        
        # Load COLMAP data
        self.cameras = read_cameras_text(os.path.join(sparse_path, "cameras.txt"))
        self.images = read_images_text(os.path.join(sparse_path, "images.txt"))
        points3D = read_points3D_text(os.path.join(sparse_path, "points3D.txt"))

        
        # Convert points3D to torch.tensor
        self.points3D_xyz = torch.as_tensor(np.array([p['xyz'] for p in points3D.values()])).float()
        self.points3D_rgb = torch.as_tensor(np.array([p['rgb'] for p in points3D.values()])).float()
        if maximum_pts_num is not None and self.points3D_xyz.shape[0] > maximum_pts_num:
            indices = torch.linspace(
                0, self.points3D_xyz.shape[0] - 1, maximum_pts_num
            ).long()
            self.points3D_xyz = self.points3D_xyz[indices]
            self.points3D_rgb = self.points3D_rgb[indices]

        # Get image paths and convert camera parameters
        self.image_paths = []
        self.camera_data = []
        
        for image_id, image_data in self.images.items():
            image_path = os.path.join(images_dir, image_data['name'])
            if os.path.exists(image_path):
                self.image_paths.append(image_path)
                camera = self.cameras[image_data['camera_id']]
                K = get_intrinsic_matrix(camera, downsample_factor)
                self.camera_data.append({
                    'K': K,
                    'R': image_data['R'],
                    't': image_data['t']
                })
                if max_views is not None and len(self.image_paths) >= max_views:
                    break
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (0,0), fx=1./self.downsample_factor, fy=1./self.downsample_factor)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.FloatTensor(image) / 255.0
        
        # Get camera parameters
        camera_data = self.camera_data[idx]
        K = torch.FloatTensor(camera_data['K'])
        R = torch.FloatTensor(camera_data['R'])
        t = torch.FloatTensor(camera_data['t'])
        
        return {
            'image': image,
            'K': K,
            'R': R,
            't': t,
            'image_path': image_path
        }
