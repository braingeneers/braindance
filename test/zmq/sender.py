import zmq

maxone = "128.111.208.217"
gpu = "128.111.209.218"

def run(ip_address=gpu, port=1150):
    print("Connecting ...")
    publisher = zmq.Context().instance().socket(zmq.PUB)
    publisher.bind(f"tcp://{ip_address}:{port}")

    while True:
        msg = input("\nWaiting for input to send: ")
        print("Sending ...")
        publisher.send(int(msg).to_bytes())
        print("Sent")
        


if __name__ == "__main__":
    run()
