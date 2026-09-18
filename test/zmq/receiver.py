import zmq

maxone = "128.111.208.217"
gpu = "128.111.209.218"

def run(ip_address=maxone, port=1150):
    print("Connecting ...")
    subscriber = zmq.Context().instance().socket(zmq.SUB)
    subscriber.setsockopt(zmq.RCVHWM, 10)  # Max msgs in buffer
    subscriber.setsockopt(zmq.RCVBUF, 10)  # Max size of msgs in buffer (bytes)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages (no filter)
    subscriber.setsockopt(zmq.RCVTIMEO, 100)  # Wait 100ms for message before timing out (raising error)
    
    subscriber.connect(f"tcp://{ip_address}:{port}")
    
    print_waiting = True
    while True:
        try:
            if print_waiting:
                print("\nWaiting for message ...")
                print_waiting = False
            msg = subscriber.recv()
            msg = int.from_bytes(msg)
            print(msg)
            print_waiting = True
        except zmq.Again:
            pass  # Continue waiting without raising an error
        
        
if __name__ == "__main__":
    run()
    