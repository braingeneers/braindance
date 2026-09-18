from braindance.core.maxwell_env import MaxwellEnv
from braindance.core.params import maxwell_params
from braindance.core.phases import CartPolePhase, PhaseManager, NeuralSweepPhase
from braindance.core.utils import SmartPlug

import numpy as np
import datetime

params = maxwell_params
params['save_dir'] = './dryrun2' # Path to the data directory, will be created if it doesn't exist
params['name'] = 'cartpole' # Name of the experiment


# Last 2 electrodes here are for the ones on the motor neurons, should be just used for tetanus
params['stim_electrodes'] =  [17189,16545, 12135, 24264,23377, 11708, 14108] # Define the 4 electrodes to stimulate here
params['max_time_sec'] = 60*30 # 30 mins

params['config'] = 'config_causal2.cfg'#'config.cfg' # Path to the config file
params['observation_type'] = 'raw'

neuron_list = np.arange(len(params['stim_electrodes']))

# Neurons to change
sensory_neurons = [0,1] # Indexes of the stim_electrodes array6
motor_neurons = [836,56] # In channels to read, [1] was 440
training_neurons= [2,3,4,5,6] # Indexes of the stim neurons

plug = SmartPlug('coleman', verbose=True)

class Experimenter:
    def __init__(self, params, neuron_list, sensory_neurons, motor_neurons, training_neurons, plug, n_episodes):
        self.params = params
        self.neuron_list = neuron_list
        self.sensory_neurons = sensory_neurons
        self.motor_neurons = motor_neurons
        self.training_neurons = training_neurons
        self.plug = plug

    def now(self):
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")


    def run_experiment(self):
        # Turn smartpulg on
        if self.plug != None:
            self.plug.turn_on()

        env = MaxwellEnv(**self.params)

        cartpole = CartPolePhase(env, sensory_neurons=self.sensory_neurons, motor_neurons=self.motor_neurons,
                                training_neurons=self.training_neurons, verbose=True,
                                read_period_ms = 200, train_period_ms = 200,
                                n_episodes=n_episodes, artifact_removal=True)
        
        phase_manager = PhaseManager(env)
        phase_manager.add_phase(cartpole)


        phase_manager.summary()

        phase_manager.run()

        if self.plug != None:
            self.plug.turn_off()



# def run_experiment(params):
#     # Turn smartpulg on
#     plug.turn_on()

#     env = MaxwellEnv(**params)


#     cartpole = CartPolePhase(env, sensory_neurons=sensory_neurons, motor_neurons=motor_neurons,
#                             training_neurons=training_neurons, verbose=True,
#                             read_period_ms = 200)

#     phase_manager = PhaseManager(env)
#     phase_manager.add_phase(cartpole)


#     phase_manager.summary()

#     phase_manager.run()

#     plug.turn_off()


import schedule
import time
if __name__ == '__main__':
    n_episodes = 500
    # plug = None
    experimenter = Experimenter(params, neuron_list, sensory_neurons, motor_neurons, training_neurons,
                     plug, n_episodes=n_episodes)

    # Schedule experiment every 2 hourse between 7pm and 9am
    schedule.every().day.at("19:00").do(experimenter.run_experiment)
    schedule.every().day.at("21:00").do(experimenter.run_experiment)
    schedule.every().day.at("23:00").do(experimenter.run_experiment)
    schedule.every().day.at("01:00").do(experimenter.run_experiment)
    schedule.every().day.at("03:00").do(experimenter.run_experiment)
    schedule.every().day.at("05:00").do(experimenter.run_experiment)
    schedule.every().day.at("07:00").do(experimenter.run_experiment)
    schedule.every().day.at("09:00").do(experimenter.run_experiment)
    schedule.every().day.at("11:00").do(experimenter.run_experiment)
    schedule.every().day.at("13:00").do(experimenter.run_experiment)
    schedule.every().day.at("15:00").do(experimenter.run_experiment)
    schedule.every().day.at("17:00").do(experimenter.run_experiment)

    print("Beginning whole experiment at " + experimenter.now())

    # Run experiment once
    experimenter.run_experiment()

    while True:
        n = schedule.idle_seconds()
        if n is None:
            break
        n = int(n)
        
        if n % (60*5) == 0 or ((n < 60*5) and (n%(60*5) == 0)): # only print every 5 minutes or every min <5mins
            hours = n // 3600
            minutes = (n % 3600) // 60
            seconds = n % 60
            print(f"Waiting {hours:02}:{minutes:02}:{seconds:02} until next job")

        schedule.run_pending()

        time.sleep(1)