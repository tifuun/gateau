"""!
@file 
File containing the threadmanager class for TiEMPO2.
This class is responsible for launching heavy calculations on a separate daemon thread,
preventing the program from becoming unresponsive and allowing for Ctrl-c keyboard escapes.
"""

import threading

class Manager(object):
    """!
    This class generates a threadmanager object.
    The manager can start daemon threads and signal when the thread is finished.
    This class is only used to spawn calls to the C++ backend inside a daemon thread so that Python keeps control over the process.
    This allows users to Ctrl-c a running calculation in C++ from Python.
    """

    def __init__(self, callback=None):
        self.callback = callback
    
    def new_thread(self, target, args):
        t = threading.Thread(target=target, args=args)
        t.daemon = True
        t.start()
    
        while t.is_alive(): # wait for the thread to exit
            t.join(0.1)

    def on_thread_finished(self):
        if self.callback is not None:
            self.callback()
