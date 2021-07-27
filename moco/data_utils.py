import numpy as np
import os
from torch.utils.data import Dataset
import json
import glob

def read_off(file_path):
    with open(file_path) as file:
        header = file.readline().strip()
        if 'OFF' != header:
            n_verts, n_faces, n_dontknow = tuple([int(float(s)) for s in header[3:].strip().split(' ')])
        else:
            n_verts, n_faces, n_dontknow = tuple([int(float(s)) for s in file.readline().strip().split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        # faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return np.array(verts)

def read_obj(file_path):
    points = []
    with open(file_path) as file:
        for line in file:
            if len(line.split()) > 0 and line.split()[0] == 'v':
                points.append([float(x) for x in line.split()[1:]])
    return np.array(points)

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class PointCloudNormalize(object):
    
    def __call__(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

class ShapeNetDataLoader(Dataset):
    def __init__(self, shapenet_path='/pasteur/data/ShapeNet', classes=['airplane', 'chair', 'sofa', 'table', 'boat', 'rifle', 'car'], transform=None, npoints=1024, normalize=False):
        # initialize stuff
        self.transform = transform
        self.npoints = npoints
        self.normalize = normalize
        
        # extract the mapping of synset IDs to class names
        json_path = os.path.join(shapenet_path, 'ShapeNetCore.v2', 'taxonomy.json')
        with open(json_path) as json_file:
            class_info = json.load(json_file)
            
        # find filenames corresponding to all models in the given classes
        self.all_files = []
        if classes is not None:
            self.classes = classes
            for i in range(len(self.classes)):
                for item in class_info:
                    if item['name'].split(',')[0] == self.classes[i]:
                        print(item['name'])
                        cls_dir = os.path.join(shapenet_path, 'ShapeNetCore.v2', item['synsetId'])
                        for path, subdirs, files in os.walk(cls_dir):
                            for name in files:
                                if name == 'model_normalized.obj':
                                    self.all_files.append((os.path.join(path, name), i))
        else:
            self.classes = []
            for item in class_info:
                print(item['name'], len(self.classes))
                cls_dir = os.path.join(shapenet_path, 'ShapeNetCore.v2', item['synsetId'])
                for path, subdirs, files in os.walk(cls_dir):
                    for name in files:
                        if name == 'model_normalized.obj':
                            self.all_files.append((os.path.join(path, name), len(self.classes))) 
                self.classes.append(item['name'])
        print(len(self.all_files))
        
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        
        path, cls = self.all_files[idx]
        
        pcloud = read_obj(path)
        
        # sample #(npoints) points without replacement from the point cloud
        if pcloud.shape[0] > self.npoints:
            pcloud = pcloud[np.random.choice(pcloud.shape[0], self.npoints, replace=False), :]
        else:
            pcloud = pcloud[np.random.choice(pcloud.shape[0], self.npoints), :]
        
        if self.normalize:
            pcloud = pc_normalize(pcloud)
        
        if self.transform:
            pcloud = self.transform(pcloud)
            
        return pcloud, cls  

def generate_random_drift(pc,sigma=0.02):
    # pc [N*3] 
    # sigma [1],for controlling the random distance
    # pc_gen [N*3]
    pc_gen = torch.zeros(0)
    for pt in pc:  
        #pt [1*3] x y z
        pt = torch.FloatTensor(pt)
        pt_new = torch.normal(pt, sigma, out=None).unsqueeze(-1).transpose(0,1)
        pc_gen = torch.cat([pc_gen, pt_new], 0)
    return pc_gen

class ShapeNet(Dataset):
    def __init__(self,root,mode='train',sigma=0.04,transform=None):
        self.sigma = sigma
        self.mode = mode
        self.transform = transform
        
        if mode=='both':
            data_train = np.load(os.path.join(root,'train','pntcloud_full.npy'))
            data_test = np.load(os.path.join(root,'test','pntcloud_full.npy'))
            self.data = np.concatenate((data_train,data_test),axis=0)
            label_train = np.load(os.path.join(root,'train','label_full.npy'))
            label_test = np.load(os.path.join(root,'test','label_full.npy'))
            self.label = np.concatenate((label_train,label_test),axis=0)
        else:
            self.data = np.load(os.path.join(root,'train','pntcloud_7.npy'))
            self.label = np.load(os.path.join(root,'train','label_7.npy'))
        
        self.train_num = self.label.shape[0]
        self.indices = range(self.data.shape[0])
        
    def __getitem__(self,index,is_online=True):
        pc_gt = self.data[index]
        sig = self.sigma
        pc = generate_random_drift(pc_gt,sigma=sig).tolist()
        pc = torch.FloatTensor(pc).transpose(0,1) 
        pc_gt = torch.FloatTensor(pc_gt).transpose(0,1)
        if self.transform:
            pc = self.transform(pc)
        return pc, pc_gt#, self.indices[index]

    def __len__(self):
        return len(self.data)
        
class MiniShapeNetDataLoader(Dataset):
    def __init__(self, shapenet_path='/pasteur/data/ShapeNet', classes=['airplane', 'chair', 'sofa', 'table', 'boat', 'rifle', 'car'], transform=None, npoints=1024, normalize=False, examples=100):
        # initialize stuff
        self.classes = classes
        self.transform = transform
        self.npoints = npoints
        self.normalize = normalize
        
        # extract the mapping of synset IDs to class names
        json_path = os.path.join(shapenet_path, 'ShapeNetCore.v2', 'taxonomy.json')
        with open(json_path) as json_file:
            class_info = json.load(json_file)
            
        # find filenames corresponding to all models in the given classes
        self.all_files = []
        for i in range(len(self.classes)):
            cls_examples = []
            for item in class_info:
                if item['name'].split(',')[0] == self.classes[i]:
                    print(item['name'])
                    cls_dir = os.path.join(shapenet_path, 'ShapeNetCore.v2', item['synsetId'])
                    for path, subdirs, files in os.walk(cls_dir):
                        for name in files:
                            if name == 'model_normalized.obj':
                                cls_examples.append((os.path.join(path, name), i))
            self.all_files = self.all_files + cls_examples[:examples]
                    
        
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        
        path, cls = self.all_files[idx]
        
        pcloud = read_obj(path)
        
        # sample #(npoints) points without replacement from the point cloud
        if pcloud.shape[0] > self.npoints:
            pcloud = pcloud[np.random.choice(pcloud.shape[0], self.npoints, replace=False), :]
        else:
            pcloud = pcloud[np.random.choice(pcloud.shape[0], self.npoints), :]
        
        if self.normalize:
            pcloud = pc_normalize(pcloud)
        
        if self.transform:
            pcloud = self.transform(pcloud)
            
        return pcloud, cls
    
def load_data_partseg(partition):
    DATA_DIR = '/pasteur/data'
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*%s*.h5'%partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg
    
class ShapeNetPart(Dataset):
    def __init__(self, num_points, partition='train', class_choice=None, transform=None):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition        
        self.class_choice = class_choice
        self.transform = transform

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            # pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        if self.transform != None:
            pointcloud = self.transform(pointcloud)
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]