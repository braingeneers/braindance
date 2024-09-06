from braindance.core.spikedetector.model import ModelSpikeSorter
import torch

try:
    import torch_tensorrt
    model_path = '../core/spikedetector/model_256ch.pt'
    print("Using TensorRT")
except ImportError:
    model_path = '../core/spikedetector/model_256ch_torch.pt'
    map_location=torch.device('cpu')
    print("Using PyTorch")





class SpikeDetector():
    def __init__(self, model_path, n_channels=256, n_frames = 200, device='cuda'):
        self.model = torch.jit.load(model_path, map_location=device)
        self.model = self.model.to('cpu').float()
        # self.data_slice = torch.zeros(n_channels,1, n_frames).half().to('cpu')#.cuda()
        self.data_slice = torch.zeros(n_channels,1, n_frames).to('cpu').float()#.cuda()
        self.ind = 0
        self.n_channels = n_channels
        self.n_frames = n_frames

    
    def detect(self, data_frame):
        '''Takes in the current frame of data, adds it to the data slice,
         and returns the spike predictions
         
         Parameters
         ----------
            data_frame : np.array of shape (n_channels, 1)
                The current frame of data
        Returns
        -------
            spike_preds : np.array of shape (n_channels, 120)
                The spike predictions for each channel, for the 
                previous 6ms, or 120 frames

         '''
        # self.data_slice[:, 0, self.ind] = torch.from_numpy(data_frame).half().cuda()
        
        # # # Iterate the index
        # if self.ind == self.n_frames - 1:
        #     self.ind = 0
        # else:
        #     self.ind += 1

        self.data_slice[:, 0, :-1] = self.data_slice[:, 0, 1:]
        self.data_slice[:, 0, -1] = torch.from_numpy(data_frame).half().to('cpu')#.cuda()
        


        # Get the predictions
        # spike_preds = self.model(self.data_slice).cpu().detach().numpy()
        # zeros instead
        spike_preds = torch.zeros(self.n_channels, 120).to('cpu').detach().numpy()
        return spike_preds
    

    def detect_chunk(self, data_chunk):
        '''
        Takes in a chunk of data, and returns the spike predictions for each
        channel for the last 6ms of data
        
        Parameters
        ----------
            data_chunk : np.array of shape (n_channels, n_frames)
                The chunk of data to be processed
        Returns
        -------
            spike_preds : np.array of shape (n_channels, 120)
                The spike predictions for each channel, for the
                previous 6ms, or 120 frames
                '''
        chunk_size = data_chunk.shape[1]
        if chunk_size > self.n_frames:
            raise ValueError("Chunk size is larger than the data slice size")
        
        # If we are going to overflow the data slice, we need to split the data
        # into two parts
        if self.ind + chunk_size > self.n_frames:
            self.data_slice[:, 0, self.ind:] = torch.from_numpy(data_chunk[:, :self.n_frames-self.ind]).half().cuda()
            self.data_slice[:, 0, :chunk_size-(self.n_frames-self.ind)] = torch.from_numpy(data_chunk[:, self.n_frames-self.ind:]).half().cuda()
        else:
        
            self.data_slice[:, 0, self.ind:self.ind+chunk_size] = torch.from_numpy(data_chunk).half().cuda()

        # Iterate the index
        if self.ind + chunk_size >= self.n_frames:
            self.ind = chunk_size - (self.n_frames - self.ind)
        else:
            self.ind += chunk_size
        
        # Get the predictions
        spike_preds = self.model(self.data_slice).cpu().detach().numpy()

        return spike_preds
    
    

    def reset(self):
        self.ind = 0
        self.data_slice = torch.zeros(self.n_channels,1, self.n_frames).half().cuda()

