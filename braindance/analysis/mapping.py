import numpy as np
import pandas as pd
import os

from braindance.analysis import data_loader
from braindance.io import open_file



class Mapping:
    def __init__(self, filepath=None, df=None, from_csv=False, channels=None):
        self.mapping = None
        self.selected_electrodes = []
        self.selected_channels = []
        if filepath:
            if from_csv:
                with open_file(filepath, 'r') as f:
                    self.mapping = pd.read_csv(f)
            else:
                self.mapping = data_loader.load_mapping_maxwell(filepath, channels)
            self.set_mapping(self.mapping)
        elif type(df) == pd.core.frame.DataFrame:
            self.mapping = df
            self.set_mapping(self.mapping)

    @classmethod
    def from_csv(cls, filepath):
        return cls(filepath, from_csv=True)
    
    @classmethod
    def from_df(cls, df):
        return cls(df=df)
    
    @classmethod
    def from_maxwell(cls, filepath, channels=None):
        return cls(filepath, channels=channels)
        
    def select_electrodes(self, electrodes):
        self.selected_electrodes = electrodes
        self.selected_channels = [self.mapping[self.mapping['electrode'] == electrode]['channel'].values[0]
                                  for electrode in electrodes]

    def select_channels(self, channels):
        self.selected_channels = channels
        self.selected_electrodes = [self.mapping[self.mapping['channel'] == channel]['electrode'].values[0]
                                    for channel in channels]

    def set_mapping(self, mapping):
        if mapping is None:
            print("No mapping provided")
            self.mapping = None
            self.channels = None
            self.electrodes = None
            return
        self.mapping = mapping
        self.channels = mapping['channel'].values
        self.channels = [int(ch) for ch in self.channels]
        self.electrodes = mapping['electrode'].values
        self.electrodes = [int(elec) for elec in self.electrodes]

    def get_electrodes(self, channels=None, orig_channels=None):
        if channels is None and orig_channels is None:
            return self.electrodes
        elif orig_channels is not None:
            channels = orig_channels
            from_key = 'orig_channel'
        elif channels is not None:
            from_key = 'channel'
        else:
            raise ValueError("Missing required argument")

        if isinstance(channels, int):
            return self.mapping[self.mapping[from_key] == channels]['electrode'].values[0]
        
        return [self.mapping[self.mapping[from_key] == channel]['electrode'].values[0] 
                for channel in channels]

    def get_channels(self, electrodes=None, orig_channels=None):
        if electrodes is None and orig_channels is None:
            return self.channels
        elif orig_channels is not None:
            values = orig_channels
            from_key = 'orig_channel'
        elif electrodes is not None:
            from_key = 'electrode'
            values = electrodes
        else:
            raise ValueError("Missing required argument")

        if isinstance(values, int):
            return self.mapping[self.mapping[from_key] == values]['channel'].values[0]
        
        return [self.mapping[self.mapping[from_key] == value]['channel'].values[0]
                for value in values]

    def get_orig_channels(self, channels=None, electrodes=None):
        if channels is not None and electrodes is not None:
            return self.orig_channels
        elif channels is not None:
            from_key = 'channel'
            values = channels
        elif electrodes is not None:
            from_key = 'electrode'
            values = electrodes
        else:
            raise ValueError("Missing required argument")

        if isinstance(values, int):
            return self.mapping[self.mapping[from_key] == values]['orig_channel'].values[0]
        
        return [self.mapping[self.mapping[from_key] == value]['orig_channel'].values[0]
                for value in values]

    def get_nearest(self, channel=None, electrode=None, n=None, distance=None):
        """Calculates the n nearest channels to the given channel or electrode.
        First gets the distances between the given channel or electrode and all other channels.
        Then returns the n channels with the smallest distances.
        If distance is provided, return up to n channels within the given distance.
        """
        if channel is None and electrode is None:
            raise ValueError("Must provide either channel or electrode")

        if channel is not None:
            pos = self.mapping[self.mapping['channel'] == channel][['x', 'y']].values[0]
        else:
            channel = self.get_channels(electrode)
            pos = self.mapping[self.mapping['channel'] == channel][['x', 'y']].values[0]

        all_positions = self.mapping[['channel', 'x', 'y']].values
        distances = np.linalg.norm(all_positions[:, 1:3] - pos, axis=1)
        
        # Create a list of (channel, distance) tuples
        channel_distances = [(all_positions[i][0], distances[i]) for i in range(len(distances))]
        
        # Filter out the original channel itself
        channel_distances = [cd for cd in channel_distances if cd[0] != channel]
        
        # Sort by distance
        channel_distances.sort(key=lambda x: x[1])
        
        if distance is not None:
            channel_distances = [cd for cd in channel_distances if cd[1] <= distance]
        
        nearest_channels = [int(cd[0]) for cd in channel_distances]
        
        if n is not None:
            return nearest_channels[:n]
        
        if electrode is not None:
            return self.get_electrodes(nearest_channels)

        return nearest_channels

    def get_positions(self, channels=None, electrodes=None):
        if channels is not None and electrodes is not None:
            raise ValueError("Cannot provide both channels and electrodes")
        elif channels is not None:
            pass
        elif electrodes is not None:
            channels = self.get_channels(electrodes)
        else:
            # Return all positions
            return self.mapping[['x', 'y']].values
        return self.mapping[self.mapping['channel'].isin(channels)][['x', 'y']].values
    
    def save(self, filepath):
        with open_file(filepath, 'w') as f:
            self.mapping.to_csv(f, index=False)
