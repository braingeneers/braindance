import os
import socket
try:
    import maxlab
except:
    pass

def get_maxwell_status():
    '''
    Get the status of the maxwell system.
    True if initialized, False otherwise.
    '''
    port = 7200 + 15
    try:
        port = int(os.environ['MXW_BASE_PORT']) + 15
    except KeyError:
        port = 7200 + 15
    except ValueError:
        port = 7200 + 15
    except:
        print("Unhandeled exception in get_maxwell_status()")
        port = 7200 + 15

    try:
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.connect(('localhost', port))
        # serversocket.sendall(b'ping')
    except:
        return False
    return True