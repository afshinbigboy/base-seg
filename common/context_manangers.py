import signal
import sys



original_sigint_handler = signal.getsignal(signal.SIGINT)


class safezone:
    interrupted = False
    
    def __init__(self, *args, **kwargs):
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def __enter__(self):
        return None

    def __exit__(self, type, value, traceback):
        if self.interrupted:
            print("Done. Terminated safely!")
            sys.exit(0)
        else:
            signal.signal(signal.SIGINT, original_sigint_handler)
            return True

    def _signal_handler(self, signal_received, frame):
        print('\nOk, I understand you want to terminate me but at this moment I am writing on something critical! Please wait a bit to complete it ...')
        self.interrupted = True
