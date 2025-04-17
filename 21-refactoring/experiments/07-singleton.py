# %%

MAIN = "__main__" == __name__

# %%

class LoggerL1:
    def __init__(self, type: str = None):
        self.type = type
        self.logs: list[str] = []
        print("Creating SingletonLoggerL1 instance")

    def log(self, message: str):
        self.logs.append(message)
        print(f"LOG: {message}")

if MAIN:
    logger1 = LoggerL1(type="debug")
    logger1.log("First message.")
    logger2 = LoggerL1(type="release")
    logger2.log("Second message.")
    print(f"Logger1 type: {type(logger1)}")
    print(f"Logger2 type: {type(logger2)}")
    print(f"Instances are the same: {logger1 is logger2}")
    print(f"Logger1 type: {logger1.type}")
    print(f"Logger2 type: {logger2.type}")
    print(f"Logger1 Logs: {logger1.logs}")
    print(f"Logger2 Logs: {logger2.logs}")

# %%
# Reference:
# https://stackoverflow.com/a/6798042/13285583

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class MessageFormatter:
    def format_message(self, msg: str) -> str:
        return f"[Formatted] {msg}"

class SingletonLoggerL2(LoggerL1, MessageFormatter, metaclass=Singleton):
    def __init__(self, type: str = None):
        super().__init__(type=type)
        print("Creating SingletonLoggerL2 instance")
    
    def log(self, message: str):
        formatted = self.format_message(message)
        super().log(formatted)
        # super(CurrentClass, self)
        print(f"LOG L2: {formatted}")
    
    def clear_logs(self):
        self.logs.clear()
        print("Logs cleared.")

if MAIN:
    logger1 = SingletonLoggerL2(type="debug")
    logger2 = SingletonLoggerL2(type="release") # Will not print "Creating..."
    print(SingletonLoggerL2.__mro__)
    logger1.log("First message.")
    logger2.log("Second message.")
    print(f"Logger1 type: {type(logger1)}")
    print(f"Logger2 type: {type(logger2)}")
    print(f"Instances are the same: {logger1 is logger2}")
    print(f"Logger1 type: {logger1.type}")
    print(f"Logger2 type: {logger2.type}")
    print(f"Logs: {logger1.logs}")
    print(f"Logs: {logger2.logs}")
    logger1.clear_logs()
    print(f"Logs: {logger1.logs}")
    print(f"Logs: {logger2.logs}")

# %%
