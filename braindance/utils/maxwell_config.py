import pandas as pd

class Config:
    def __init__(self, filename, is_content=False):
        self.config = []
        self.mappings = []

        self.stim_electrodes = []

        if filename is None:
            print('No config file specified')
            return
        
        if is_content:
            # filename is a file-like object containing the content
            content = filename.read()
        else:
            with open(filename, 'r') as file:
                content = file.read()

        # Parse the content
        self.config = [m.replace('(', ' ').replace(')', ' ').replace('/', ' ').split() for m in content.split(';')[:-1]]
        self.mappings = [self.Mapping(*m) for m in self.config]
        self.config = [(int(m[0]), int(m[1]), float(m[2]), float(m[3])) for m in self.config]

        self.df = None
        # Set the dataframe
        self.set_df()

    def get_channels(self):
        return [m.channel for m in self.mappings]
    
    def get_electrodes(self):
        return [m.electrode for m in self.mappings]

    def get_channels_for_electrodes(self, electrodes):
        return [m.channel for m in self.mappings if m.electrode in electrodes]
    
    def get_num_channels(self):
        return len(self.get_channels())
    
    def add_stim_electrodes(self, stim_electrodes):
        # This method assumes stim_electrodes is a list of integers representing electrode IDs
        for electrode in stim_electrodes:
            if electrode not in self.get_electrodes():
                print(f'Electrode {electrode} not found in config file')
                continue
            if electrode in self.stim_electrodes:
                print(f'Electrode {electrode} already in stim_electrodes')
                continue
            self.stim_electrodes.append(electrode)
        return
    
    def set_df(self):
        """Set channel, electrode, x, y dataframe"""
        self.df = pd.DataFrame(self.config, columns=['channel', 'electrode', 'x', 'y'])
        return

    class Mapping:
        def __init__(self, channel, electrode, x, y):
            self.channel = int(channel)
            self.electrode = int(electrode)
            self.x = float(x)
            self.y = float(y)