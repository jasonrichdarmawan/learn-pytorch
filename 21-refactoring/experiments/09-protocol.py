# %%

from typing import Protocol, TypeVar, Generic
from dataclasses import dataclass

# the key best practice is: a Protocol describes what it needs
# a service depends on Protocol, not against the concrete class.
class FileLike(Protocol):
    name: str
    size: int

TFile = TypeVar("TFile", bound=FileLike)

class FileFactory(Protocol[TFile]):
    def create(self, name: str, size: int) -> TFile: ...

class FileHostingService(Generic[TFile]):
    def __init__(self, file_factory: FileFactory[TFile]):
        self._file_factory = file_factory

    def upload(self, name: str, size: int) -> TFile:
        file = self._file_factory.create(name, size)
        print(f"Uploaded file: {file.name} of size {file.size} bytes")
        return file
    
@dataclass
class FileL1(FileLike):
    name: str
    size: int
    
class FileFactoryL1(FileFactory[FileL1]):
    def create(self, name: str, size: int) -> FileL1:
        return FileL1(name, size)

@dataclass
class FileL2(FileLike):
    name: str
    size: int
    timestamp: int

class FileFactoryL2(FileFactory[FileL2]):
    def create(self, name: str, size: int, timestamp: int) -> FileL2:
        return FileL2(name, size, timestamp)
    
class FileHostingServiceL2(FileHostingService[FileL2]):
    def upload(self, name: str, size: int, timestamp: int) -> FileL2:
        file = self._file_factory.create(name, size, timestamp)
        print(f"Uploaded file: {file.name} of size {file.size} bytes at {file.timestamp}")
        return file
    
if __name__ == "__main__":
    factory_l1 = FileFactoryL1()
    service_l1 = FileHostingService(factory_l1)
    service_l1.upload("file1.txt", 1024)

    factory_l2 = FileFactoryL2()
    service_l2 = FileHostingServiceL2(factory_l2)
    service_l2.upload("file2.txt", 2048, 1625077800)

# %%