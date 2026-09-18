from pathlib import Path
import time

from pandas import DataFrame
import zmq


class BaseEnv:
    def __init__(self) -> None:
        self.save_dir = "test"

    def time_elapsed(self):
        return 0


local_ip = "127.0.0.1"
maxone_ip = "128.111.208.217"
gpu_ip = "128.111.209.218"
photon_ip = "128.111.209.54"

STOP_COMMAND = "STOP COMMAND_RECEIVER"


class CommandSender:
    """
    Class to send commands to another process with ZMQ and wait for it to finish
    
    Designed to be a wrapper of BaseEnv
    """

    def __init__(self, env: BaseEnv, log_save_name='2p_command_log.csv',
                 ip_address=maxone_ip, port=1150, setup_time_sec=2,
                 verbose=True):
        self.env = env
        self.log_save_path = Path(env.save_dir) / log_save_name

        start_socket_pub = zmq.Context().instance().socket(zmq.PUB)
        start_socket_pub.bind(f"tcp://{ip_address}:{port}")

        print(f"Waiting {setup_time_sec} seconds for socket tcp://{ip_address}:{port} to bind ...")
        time.sleep(setup_time_sec)  # Wait for socket to bind

        self.start_socket_pub = start_socket_pub

        # done_socket_sub = context.socket(zmq.SUB)  # For when 2p command is done
        self.command_log = DataFrame(columns=["command", "start_second", "end_second"])

        self.verbose = verbose

    def send(self, command, duration_sec):
        """
        Send command
        """
        if self.verbose:
            print(f"\nSending '{command}' ...")
        start = self.env.time_elapsed()
        self.start_socket_pub.send_string(command)

        if self.verbose:
            print(f"Waiting {duration_sec} seconds for command to finish ...")
        time.sleep(duration_sec)
        end = self.env.time_elapsed()

        self.command_log.loc[len(self.command_log)] = command, start, end

    def close(self):
        """
        Stops the CommandSender object by 
            Sending stop command to command_receiver
            Closing sockets
            Saving command log
        """
        self.start_socket_pub.send_string(STOP_COMMAND)
        self.start_socket_pub.close()

        self.log_save_path.parent.mkdir(exist_ok=True, parents=True)
        self.command_log.to_csv(self.log_save_path, index=False)


def command_receiver(ip_address=maxone_ip,
                     start_port=1150,
                     verbose=True):
    """
    Receive commands from another process and then start the command,
    sending a signal when done
    
    `job` must be a function with one argument: a string that contains the command from CommandSender
        and one return value: a string giving the result
        The function will process the command
    """
    print("Connecting to Prairie Link ...")
    import win32com.client
    pl = win32com.client.Dispatch("PrairieLink.Application")
    pl.Connect()

    start_socket_sub = zmq.Context().instance().socket(zmq.SUB)
    start_socket_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    start_socket_sub.connect(f"tcp://{ip_address}:{start_port}")

    while True:
        if verbose:
            print("\nWaiting for message ...")

        while True:
            try:
                command = start_socket_sub.recv_string()
                break
            except zmq.Again:
                pass

        if verbose:
            print(f"Received '{command}'")

        if command == STOP_COMMAND:
            if verbose:
                print("Stopping ...")
            start_socket_sub.close()
            pl.Disconnect()
            return

        if verbose:
            print(f"Starting '{command}'")

        # Returns when command is sent, not when command finishes
        command_sent = pl.SendScriptCommands(command)
        if not command_sent:
            print("ERROR starting command on 2-photon microscope with Prairie Link.\nMake sure that you entered a valid command")


def main():
    env = BaseEnv()
    command_sender = CommandSender(env)
    while True:
        msg = input("\nWaiting for input to send: ")
        if msg == "stop":
            command_sender.close()
            break
        command_sender.send(msg, 3)

        command_sender.send("-ZSeries", 300)


if __name__ == "__main__":
    main()

    # test = Test()
    # while True:
    #     msg = input("\nWaiting for input to send: ")
    #     print("Sending ...")
    #     # publisher.send(int(msg).to_bytes(length=1, byteorder="little"))
    #     test.send(msg)
    #     print("Sent")
    # test.send()

    # print("Connecting ...")
    # publisher = zmq.Context().instance().socket(zmq.PUB)
    # publisher.bind(f"tcp://{maxone_ip}:{1150}")

    # while True:
    #     msg = input("\nWaiting for input to send: ")
    #     print("Sending ...")
    #     # publisher.send(int(msg).to_bytes(length=1, byteorder="little"))
    #     publisher.send_string(msg)
    #     print("Sent")

    # test = input()
    # if test == "0":
    #     env = BaseEnv()
    #     sender = CommandSender(env, ip_address=local_ip)
    #     while True:
    #         msg = input("\nWaiting for input to send: ")
    #         if msg == "stop":
    #             sender.close()
    #             break
    #         sender.send(msg)
    # else:
    #     command_receiver(ip_address=local_ip)
