import numpy as np
from Layers import Base

class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.max_indices = None
        self.input_shape = None
        
    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        output = []
        overall_max_indices = []
        for i in range(input_tensor.shape[0]):
            batch = input_tensor[i,...]
            batch_output = []
            batch_max_indices = []
            for j in range(batch.shape[0]):
                channel = batch[j,...]
                output_shape = (((channel.shape[0]-self.pooling_shape[0])//self.stride_shape[0]) + 1, ((channel.shape[1]-self.pooling_shape[1])//self.stride_shape[1]) + 1)
                output_channel = np.zeros(output_shape, dtype=float)
                max_indices = np.zeros(output_channel.shape + (2,))
                for l in range(output_shape[0]):
                    for m in range(output_shape[1]):
                        window = channel[l*self.stride_shape[0]:(l*self.stride_shape[0])+self.pooling_shape[0], m*self.stride_shape[1]:(m*self.stride_shape[1])+self.pooling_shape[1]]
                        output_channel[l,m] = window.max()
                        ind = np.unravel_index(window.argmax(), window.shape)
                        max_indices[l,m,:] = (l*self.stride_shape[0]+ind[0],m*self.stride_shape[1]+ind[1])
                batch_output.append(output_channel)
                batch_max_indices.append(max_indices)
            output.append(np.stack(batch_output, axis=0))
            overall_max_indices.append((np.stack(batch_max_indices, axis=0)))
        self.max_indices = np.stack(overall_max_indices, axis=0)
        return np.stack(output,axis=0)
    
    def backward(self, error_tensor):
        output_tensor = np.zeros(self.input_shape, dtype=float)
        for i in range(error_tensor.shape[0]):
            batch = error_tensor[i,...]
            for j in range(batch.shape[0]):
                channel = batch[j,...]
                for l in range(channel.shape[0]):
                    for m in range(channel.shape[1]):
                        max_x, max_y = self.max_indices[i,j,l,m,:]
                        output_tensor[i,j,int(max_x),int(max_y)] += channel[l,m]
        return output_tensor