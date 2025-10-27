# %%

import sys
import logging

# %%

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, level, echo=None):
        self.logger = logger
        self.level = level
        self.echo = echo

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())
        if self.echo is not None:
            self.echo.write(buf)

    def flush(self):
        if self.echo is not None:
            self.echo.flush()

# %%

def running_in_jupyter():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

# %%

orig_stdout, orig_stderr = sys.stdout, sys.stderr

# %%

if not running_in_jupyter():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
        filename="output.log",
        filemode="w",
    )
    log = logging.getLogger("foobar")
    sys.stdout = StreamToLogger(logger=log, level=logging.INFO, echo=orig_stdout)
    sys.stderr = StreamToLogger(logger=log, level=logging.ERROR, echo=orig_stderr)

# %%

print("Console & log file: stdout")

# %%

raise Exception("Console & log file: stderr")

# %%

if not running_in_jupyter():
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr

# %%

print("Console: stdout")

# %%

raise Exception("Console: stderr")

# %%
