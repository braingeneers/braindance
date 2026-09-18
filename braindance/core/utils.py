import os
import uuid
try:
    from braingeneers.iot import *
except:
    print("Could not import braingeneers.iot")
    print("Please install braingeneers.iot from Braingeneerspy")
    
import time


# class Scheduler:



class SmartPlug:
    def __init__(self, smartplug="smartplug_1", verbose=False):
        self.is_on = False
        self.verbose = verbose
        if smartplug is not None:
            self.smartplug = smartplug.lower()
        else:
            self.smartplug = "none"
        self.drew = None

        if smartplug.lower() == "drew":
            try:
                import board
                import digitalio
            except:
                print("Could not import board or digitalio")
                print("Please install Adafruit-Blinka from Adafruit")
                print("Will NOT be able to use 'drew' option")

            self.mb = None
            
            self.drew = digitalio.DigitalInOut(board.C0)
            self.drew.direction = digitalio.Direction.OUTPUT
            # pass
        elif smartplug.lower() == "none":
            # Will just turn server on/off
            self.mb = None
        else:
            self.mb = MessageBroker( str(uuid.uuid4()) )
        
    def turn_on( self ):
        if self.verbose:
            print(f"Turning on {self.smartplug}.")        # set up iot

        if self.mb is not None:
            self.mb.publish_message(topic=f"smartplug/telemetry/{self.smartplug}/cmnd/Power", message="ON" )  # turn on smart plug
        elif self.smartplug == "drew":
            self.drew.value = True
        else:
            if self.verbose:
                print("Using no smartplug, just turning on server")
            #os.system("/home/mxwbio/MaxLab/bin/mxwserver.sh &")               # turn on maxwell server
        time.sleep(12)                                                     # wait 7 seconds for smart plug
        os.system("/home/mxwbio/MaxLab/bin/mxwserver.sh &")               # turn on maxwell server
        time.sleep(12)                                                     #add additional 7 seconds just in case
        self.is_on = True


    def turn_off( self ):
        
        if self.verbose:
            print(f"Turning off {self.smartplug}.")
        # set up iot
        # os.system("/home/mxwbio/MaxLab/bin/killall.sh")                         # shut down MaxOne server
        os.system("killall mxwserver")
        if self.mb is not None:
            self.mb.publish_message(topic=f"smartplug/telemetry/{self.smartplug}/cmnd/Power", message="OFF" )       # turn off smart plug
        elif self.smartplug == "drew":
            self.drew.value = False
        else:
            if self.verbose:
                print("Using no smartplug, just turning off server")

            
        
        self.is_on = False


     