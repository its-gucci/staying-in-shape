import numpy as np
from scipy.interpolate import Rbf
import torch

class UniformOrthogonal(object):
    
    def __init__(self, n, k, seed=None):
        self.n = n
        self.k = k
        if seed:
            np.random.seed(seed)
            
    def __call__(self, pcloud):
        A = np.random.standard_normal((self.n, self.k))
        Q, R = np.linalg.qr(A)
        return np.matmul(pcloud, Q)
    
class RandomRIP(object):
    
    def __init__(self, n, k, T, eps):
        self.n = n
        self.k = k
        self.T = T
        self.eps = eps
    
    def __call__(self, pcloud):
        Q = np.random.normal(loc=0, scale=1/self.n, size=(self.n, self.k))
        random_columns = np.random.randint(0, self.k, self.T)
        Q = Q.T[random_columns].T
        while np.linalg.norm(np.matmul(Q.T, Q) - np.identity(self.T)) > self.eps:
            Q = np.random.normal(loc=0, scale=1/self.n, size=(self.n, self.k))
            random_columns = np.random.randint(0, self.k, self.T)
            Q = Q.T[random_columns].T
        return np.matmul(pcloud, Q)
    
class RandomPermutation(object):
        
    def __call__(self, pcloud):
        np.random.shuffle(pcloud)
        return pcloud
    
class RandomPerturbationUniform(object):
    
    def __init__(self, eps):
        self.eps = eps
        
    def __call__(self, pcloud):
        return pcloud + np.random.uniform(-self.eps, self.eps, size=pcloud.shape)
    
class RandomPerturbationGaussian(object):
    
    def __init__(self, eps):
        self.eps = eps
        
    def __call__(self, pcloud):
        return pcloud + np.random.normal(0, self.eps, size=pcloud.shape)

class RandomRbf(object):
    
    def __init__(self, mean, sigma, size):
        self.mean = mean
        self.sigma = sigma
        self.size = size
        
    def __call__(self, pcloud):
        new = np.zeros(pcloud.shape)
        mins = np.min(pcloud, axis=0)
        maxes = np.max(pcloud, axis=0)
        for i in range(3):
            x = np.random.uniform(mins[0], maxes[0], size=self.size)
            y = np.random.uniform(mins[1], maxes[1], size=self.size)
            z = np.random.uniform(mins[2], maxes[2], size=self.size)
            d = np.random.normal(loc=self.mean, scale=self.sigma, size=self.size)
            deform = Rbf(x, y, z, d)
            new[:, i] = pcloud[:, i] + deform(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2])
        return new
    
class NumpyToTensor(object):
    
    def __call__(self, arr):
        return torch.from_numpy(arr)
    
def softmax(x, temp):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x))/temp)
    return e_x / e_x.sum()
    
class SmoothingOp(object):
    
    def __init__(self, k, temp=1):
        self.k = k
        self.temp = temp
    
    def __call__(self, pc):
        for i in range(len(pc)):
            dists = np.sum(np.square(pc - pc[i]), axis=1)
            knearest = np.argpartition(dists, self.k+1)[:self.k+1]
            pc[i] = np.sum(np.expand_dims(softmax(dists[knearest], self.temp), axis=1) * pc[knearest], axis=0)
        return pc
    
class RandomApply(object):
    
    def __init__(self, transformations, p=None):
        self.transformations = transformations
        if p is not None:
            self.p = p
        else:
            self.p = np.full(len(transformations), 1/len(transformations))
        assert len(self.transformations) == len(self.p)
        
    def __call__(self, pc):
        r = np.random.choice(range(len(self.transformations)), p=self.p)
        return self.transformations[r](pc)

# random p-rotation from Isometry Robustness paper
class rotate_all(object):
    def __init__(self, p=0.5, rotate_range = None):
        if rotate_range is None:
            self.rotate_range = [0, np.pi*2]
        self.p = p
        
    def __call__(self, pointcloud):
        if np.random.uniform() < self.p:
            angles = np.random.uniform(low = self.rotate_range[0], high = self.rotate_range[1], size=3)
            cos1, sin1 = np.cos(angles[0]), np.sin(angles[0])
            cos2, sin2 = np.cos(angles[1]), np.sin(angles[1])
            cos3, sin3 = np.cos(angles[2]), np.sin(angles[2])
            rotation_matrix = np.array([[cos1*cos3-cos2*sin1*sin3, -cos2*cos3*sin1-cos1*sin3,  sin1*sin2],
                                        [cos3*sin1+cos1*cos2*sin3,  cos1*cos2*cos3-sin1*sin3, -cos1*sin2],
                                        [      sin2*sin3         ,       cos3*sin2          ,    cos2   ]])

            pointcloud = np.dot(pointcloud, rotation_matrix)#.astype(np.float32)
        return pointcloud
    
class y_rotate(object):
    
    def __call__(self, pointcloud):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        return np.matmul(pointcloud, rotation_matrix)
    
class rotate_by_deg(object):
    def __init__(self, deg, p=1):
        self.angle = deg * np.pi/180
        self.p = p
        
    def __call__(self, pointcloud):
        if np.random.uniform() < self.p:
            cos1, sin1 = np.cos(self.angle), np.sin(self.angle)
            cos2, sin2 = np.cos(self.angle), np.sin(self.angle)
            cos3, sin3 = np.cos(self.angle), np.sin(self.angle)
            rotation_matrix = np.array([[cos1*cos3-cos2*sin1*sin3, -cos2*cos3*sin1-cos1*sin3,  sin1*sin2],
                                        [cos3*sin1+cos1*cos2*sin3,  cos1*cos2*cos3-sin1*sin3, -cos1*sin2],
                                        [      sin2*sin3         ,       cos3*sin2          ,    cos2   ]])

            pointcloud = np.dot(pointcloud, rotation_matrix).astype(np.float32)
        return pointcloud