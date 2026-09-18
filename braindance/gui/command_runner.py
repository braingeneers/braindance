import subprocess
import os
import signal
from PyQt5.QtCore import QThread, pyqtSignal

class CommandRunner(QThread):
    output = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, command):
        super().__init__()
        self.command = command
        self.process = None

    def run(self):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.process = subprocess.Popen(
            self.command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
            preexec_fn=os.setsid
        )

        while True:
            # Read all available lines from the output
            while True:
                line = self.process.stdout.readline()
                if not line:
                    break # No more lines to read
                self.output.emit(line.strip())

            # Check if process has terminated
            if self.process.poll() is not None:
                if self.process.returncode != 0:
                    self.error.emit(f"Command exited with error code {self.process.returncode}")
                else:
                    self.finished.emit()
                break

    def stop(self):
        if self.process.poll() is None: # Check if the process is still running
            os.killpg(os.getpgid(self.process.pid), signal.SIGINT) # Send a Ctrl+C interrupt signal to the process group

            try:
                self.process.wait(timeout=5) # Wait for the subprocess to finish (with a timeout)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM) # If the subprocess doesn't finish within the timeout, send a termination signal to the process group

                try:
                    self.process.wait(timeout=2) # Wait for the subprocess to finish (with a shorter timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL) # If the subprocess still doesn't finish, force kill the process group

            self.terminate() # Terminate the QThread
            self.wait() # Wait for the QThread to finish