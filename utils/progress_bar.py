import time
import sys


class Bar:
    def __init__(self, max_step, name):
        self.max_step = max_step
        self.count = 0
        self.name = name

    def start(self):
        sys.stdout.write("[%s: %s>]" % (self.name, " " * self.max_step))
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.max_step+1)) # return to start of line, after '['

    def step(self):
        self.count += 1
        # update the bar
        sys.stdout.write("=")
        sys.stdout.flush()
        if self.count == self.max_step:
            sys.stdout.write("] Finished.\n")


if __name__ == '__main__':
    b = Bar(40, 'lookup table progress')
    b.start()
    for i in range(40):
        time.sleep(0.1)
        b.step()
