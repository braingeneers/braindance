DEBUG = False

class Sequence:
    def __init__(self):
        if DEBUG:
            print('Sequence created')
        self.sequence = []

    def append(self, item):
        self.sequence.append(item)

    def __getitem__(self, key):
        return self.sequence[key]

    def __len__(self):
        return len(self.sequence)

    def __iter__(self):
        return iter(self.sequence)

    def __str__(self):
        return str(self.sequence)

    def send(self):
        if DEBUG:
            print(self.sequence)
        return self.sequence

class Core:
    def enable_stimulation_power(self, bool_value):
        if DEBUG:
            print(bool_value)
        return "Stimulation Power Set"

class Amplifier:
    def set_gain(self, gain):
        if DEBUG:
            print(gain)
        return "Gain Set"

class unit:
    def power_up(self, do_power_up):
        if DEBUG:
            print("Power Up: ", do_power_up)
        return self
    
    def connect(self, do_connect):
        if DEBUG:
            print("Connected: ", do_connect)
        return self
    
    def set_voltage_mode(self):
        if DEBUG:
            print("Voltage Mode")
        return self
    
    def dac_source(self, source):
        if DEBUG:
            print("DAC Source: ", source)
        return self

class chip:
    @staticmethod
    def DAC(channel, value):
        if DEBUG:
            print(f"DAC,{channel},{value}")
        return f"DAC,{channel},{value}"

    @staticmethod
    def DelaySamples(value):
        if DEBUG:
            print(f"DelaySamples,{value}")
        return f"DelaySamples,{value}"

    @staticmethod
    def StimulationUnit(int_value):
        if DEBUG:
            print(int_value)
        return unit()

    class Core:
        @staticmethod
        def enable_stimulation_power(bool_value):
            if DEBUG:
                print(bool_value)
            return "Stimulation Power Set"
        
    class Array:
        def __init__(self, name):
            if DEBUG:
                print('Array', name, 'created')
            return
        def select_stimulation_electrodes(self, electrode_list):
            if DEBUG:
                print(electrode_list)
            return "Electrodes Selected"

        def connect_electrode_to_stimulation(self, electrode):
            if DEBUG:
                print(electrode)
            return "Electrode Connected"

        def query_stimulation_at_electrode(self, electrode):
            if DEBUG:
                print(electrode)
            return 1

        def load_config(self, config):
            if DEBUG:
                print(config)
            return "Config Loaded"

        def reset(self):
            if DEBUG:
                print("Reset")
            return "Reset Done"
        def download(self):
            if DEBUG:
                print("Download")
            return "Download Done"

    class Amplifier:
        @staticmethod
        def set_gain(gain):
            if DEBUG:
                print(gain)
            return "Gain Set"

    @staticmethod
    def send(value):
        if DEBUG:
            print(value)
        return value

    @staticmethod
    def send_raw(value):
        if DEBUG:
            print(value)
        return value


class saving:
    class Saving:
        def open_directory(self, save_dir):
            if DEBUG:
                print(f"Opening directory: {save_dir}")
            return "Directory Opened"

        def set_legacy_format(self, format):
            if DEBUG:
                print(f"Setting legacy format: {format}")
            return "Format Set"

        def group_delete_all(self):
            if DEBUG:
                print("Deleting all groups")
            return "Groups Deleted"

        def group_define(self, group_id, group_name):
            if DEBUG:
                print(f"Defining group: {group_id}, {group_name}")
            return "Group Defined"

        def start_file(self, file_name):
            if DEBUG:
                print(f"Starting file: {file_name}")
            return "File Started"
        
        def start_recording(self, group):
            if DEBUG:
                print("Starting recording, group:", group)
            return "Recording Started"
        
        def stop_recording(self):
            if DEBUG:
                print("Stopping recording")
            return "Recording Stopped"
        
        def stop_file(self):
            if DEBUG:
                print("Stopping file")
            return "File Stopped"
        

class system:
    @staticmethod
    def DelaySamples(value):
        if DEBUG:
            print('Delaying',value)
        return value
    
    @staticmethod
    def Event(*args):
        if DEBUG:
            print('Event',args)
        return args

class util:
    def offset():
        if DEBUG:
            print("Offset function called")
        return

    def initialize():
        if DEBUG:
            print("Initialize function called")
        return
    
    def set_gain(gain):
        if DEBUG:
            print("Set Gain function called",gain)
        return

    def hpf():
        if DEBUG:
            print("HPF function called")
        return
    
def send(value):
        if DEBUG:
            print(value)
        return value

def send_raw(value):
    if DEBUG:
        print(value)
    return value
