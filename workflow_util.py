import sys


class dummy_out:

    def write(self, data):
        pass


class block_stdout:

    def __enter__(self):
        self.old_target, sys.stdout = sys.stdout, dummy_out()

    def __exit__(self, type, value, traceback):
        sys.stdout = self.old_target
