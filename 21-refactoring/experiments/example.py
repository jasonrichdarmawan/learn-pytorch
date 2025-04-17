# %%

from typing import Dict
import heapq
from typing import Protocol, TypeVar, Generic, Optional

MAIN = "__main__" == __name__

# %% Level 1

class FileL1:
    def __init__(self, file_name: str, size: int):
        self.file_name = file_name
        self.size = size

    def __repr__(self):
        return f"FileL1({self.file_name=}, {self.size=})"

FL1 = TypeVar('FL1', bound=FileL1)

class Directory(Generic[FL1]):
    def __init__(self):
        self.files: Dict[str, FL1] = {}
        self.subdirectories: Dict[str, Directory[FL1]] = {}

class FileFactory(Protocol[FL1]):
    def __call__(self, *args, **kwargs) -> FL1: ...

class FileFactoryL1(FileFactory[FileL1]):
    def __call__(self, file_name: str, size: int) -> FileL1:
        print(f"Creating FileL1 instance: {file_name} ({size} bytes)")
        return FileL1(file_name, size)

FF1 = TypeVar('FF1', bound=FileFactoryL1)

class FileHostingServiceL1(Generic[FL1, FF1]):
    def __init__(self, file_factory: FF1):
        self.root = Directory[FL1]()
        self._file_factory = file_factory

    def _resolve_path(self, file_name: str) -> tuple[Directory[FL1], str]:
        parts = file_name.split("/")
        dir_parts, file_part = parts[:-1], parts[-1]
        current_dir = self.root
        for part in dir_parts:
            if part not in current_dir.subdirectories:
                current_dir.subdirectories[part] = Directory[FL1]()
            current_dir = current_dir.subdirectories[part]
        return current_dir, file_part

    def _check_pre_upload(self, current_dir: Directory[FL1], file_part: str) -> None:
        if file_part in current_dir.files:
            raise RuntimeError(f"File {file_part} already exists (L1 check).")

    def _store_file(self, current_dir: Directory[FL1], file_part: str, file: FL1):
        current_dir.files[file_part] = file
        print(f"Uploaded {file_part} ({file.size} bytes)")
    
    def file_upload(self, file_name: str, size: int):
        """
        - Upload the file to the remote storage server
        - If a file with the same name already exists on the server, throws a runtime exception
        """
        current_dir, file_part = self._resolve_path(file_name)

        # If a file with the same name already exists on the server, throws a runtime exception
        self._check_pre_upload(current_dir, file_part)
        
        factory_args = {"file_name": file_part, "size": size}
        file_obj = self._file_factory(**factory_args)
        
        # Upload the file to the remote storage server
        self._store_file(current_dir, file_part, file_obj)

    def file_get(self, file_name: str) -> Optional[int]:
        """
        - Returns size of the file, or nothing if the file doesn't exist
        """
        current_dir, file_part = self._resolve_path(file_name)

        # Returns size of the file, or nothing if the file doesn't exist
        file = current_dir.files.get(file_part, None)
        if file is None:
            print(f"File {file_name} does not exist.")
            return None
        return file.size
    
    def _check_pre_copy(self, source_dir: Directory, source_part: str, 
                        dest_dir: Directory, dest_part: str):
        if source_part not in source_dir.files:
            # If the source file doesn't exist, throws a runtime exception
            raise RuntimeError(f"Source file {source_part} does not exist.")
        
        # Copy the source file to a new location
        if dest_part in dest_dir.files:
            # If the destination file already exists, overviews the existing file
            print(f"Warning: Overwriting existing file {dest_part}.")

    def file_copy(self, source: str, dest: str):
        """
        - Copy the source file to a new location
        - If the source file doesn't exist, throws a runtime exception
        - If the destination file already exists, overwrites the existing file
        """
        source_dir, source_part = self._resolve_path(source)
        dest_dir, dest_part = self._resolve_path(dest)

        self._check_pre_copy(source_dir, source_part, dest_dir, dest_part)
        dest_dir.files[dest_part] = source_dir.files[source_part]
        print(f"Copied {source} to {dest}")

if MAIN:
    factory = FileFactoryL1()
    service = FileHostingServiceL1[FileL1, FileFactoryL1](file_factory=factory)
    service.file_upload("file-1.txt", 2)
    service.file_upload("file-2.txt", 1)
    assert service.file_get("file-1.txt") == 2
    assert service.file_get('non_existent.file') == None
    assert service.file_get("file-2.txt") == 1
    service.file_copy("file-1.txt", "file-1-copy.txt")
    assert service.file_get("file-1-copy.txt") == 2
    try:
        service.file_upload("file-1.txt", 5000) # This will raise RuntimeError
    except RuntimeError as e:
        print(f"Caught expected error: {e}")
    try:
        service.file_copy("unknown.txt", "new.txt") # This will raise RuntimeError
    except RuntimeError as e:
        print(f"Caught expected error: {e}")

    service.file_upload("dir-a/dir-c/file-2.txt", 1)
    try:
        service.file_upload("dir-a/dir-c/file-2.txt", 1)
    except RuntimeError as e:
        print(f"Caught expected error: {e}")
    assert service.file_get("dir-a/dir-c/file-2.txt") == 1
    service.file_copy("dir-a/dir-c/file-2.txt", "dir-a/dir-c/file-2-copy.txt")
    assert service.file_get("dir-a/dir-c/file-2-copy.txt") == 1

# %% Level 2

class FileHostingServiceL2(FileHostingServiceL1[FL1, FF1]):
    def __init__(self, file_factory):
        super().__init__(file_factory=file_factory)

    def file_search(self, prefix: str) -> list[FL1]:
        """
        - Find top 10 files starting with the provided prefix. Order results by their size in descending order, and in case of a tie by file name.
        """
        current_dir, file_part = self._resolve_path(prefix)
    
        # version 1: no heapq
        # matching_files = [
        #     file for file_name, file in current_dir.files.items()
        #     if file_name.startswith(file_part)
        # ]
        # matching_files.sort(key=lambda file: (-file.size, file.file_name))
        # return matching_files[:10]
    
        # version 2
        matching_files: list[tuple[int, str]] = []
        for file_name, file in current_dir.files.items():
            if file_name.startswith(file_part):
                heapq.heappush(matching_files, (-file.size, file_name, file))
        # Get the top 10 results
        top_10: list[FL1] = []
        count = 0
        while matching_files and count < 10:
            _, _, file = heapq.heappop(matching_files)
            # heappop returns the smallest item
            top_10.append(file)
            count += 1
        return top_10
    
if MAIN:
    factory = FileFactoryL1()
    service = FileHostingServiceL2[FileL1, FileFactoryL1](file_factory=factory)
    service.file_upload("file-1.zip", 2)
    service.file_upload("file-2.txt", 1)
    service.file_upload("file-3.log", 3)
    service.file_upload("file-4.csv", 4)
    service.file_upload("file-5.log", 5)
    service.file_upload("file-6.mdx", 6)
    service.file_upload("file-7.log", 7)
    service.file_upload("file-8.txt", 8)
    service.file_upload("file-9.txt", 9)
    service.file_upload("file-10.txt", 10)
    service.file_upload("file-11.txt", 11)
    service.file_upload("blackberry.txt", 95)
    service.file_upload("avocado.txt", 200)

    print(f"Search for 'file': {service.file_search('file')}")
    print(f"Search for 'filt': {service.file_search('filt')}")
    print(f"Search for 'a': {service.file_search('a')}")
    print(f"Search for 'b': {service.file_search('b')}")
    print(f"Search for 'c': {service.file_search('c')}")

# %% Level 3

class FileL2(FileL1):
    """
    Files now might have a specified time to live on the server.
    Implement extensions of existing methods which inherit all functionality but also with an additional parameter to incldue a timestamp for the operation,
    and new files might specify the time to live
    - no ttl means lifetime being infinite.
    """

    def __init__(self, file_name: str, size: int, timestamp: Optional[int] = None, ttl: Optional[int] = None):
        self.timestamp = timestamp
        super().__init__(file_name, size)
        self.ttl = ttl

    def __repr__(self):
        return f"FileL2({self.file_name=}, {self.size=}, {self.timestamp=}, {self.ttl=})"

    def is_alive(self, timestamp: int) -> bool:
        if self.ttl is None:
            return True
        return (self.timestamp + self.ttl) > timestamp

class FileFactoryL2(FileFactory[FileL2]):
    def __call__(self, file_name: str, size: int, timestamp: Optional[int] = None, ttl: Optional[int] = None) -> FileL2:
        print(f"Creating FileL2 instance: {file_name} ({size} bytes) with timestamp {timestamp} and ttl {ttl}")
        return FileL2(file_name=file_name, size=size, timestamp=timestamp, ttl=ttl)

FL2 = TypeVar('FL2', bound=FileL2)
FF2 = TypeVar('FF2', bound=FileFactoryL2)

class FileHostingServiceL3(FileHostingServiceL2[FL2, FF2]):
    def __init__(self, file_factory):
        super().__init__(file_factory=file_factory)

    def file_upload_at(self, timestamp: float, file_name: str, file_size: int, ttl: Optional[int] = None):
        """
        - If ttl is provided, the uploaded file is available for ttl seconds.
        """
        current_dir, file_part = self._resolve_path(file_name)

        self._check_pre_upload(current_dir, file_part)

        factory_args = {"file_name": file_part, "size": file_size, "timestamp": timestamp, "ttl": ttl}
        file_obj = self._file_factory(**factory_args)

        self._store_file(current_dir, file_name, file_obj)

    def file_get_at(self, timestamp: float, file_name: str) -> Optional[int]:
        current_dir, file_part = self._resolve_path(file_name)

        file = current_dir.files.get(file_name, None)
        if file is None:
            print(f"File {file_name} does not exist.")
            return None
        if file.is_alive(timestamp):
            print(f"File {file_name} is alive at timestamp {timestamp}.")
            return file.size
        return None
    
    def _check_pre_copy_l3(self, timestamp: float, 
                           source_dir: Directory[FL2], source_part: str, 
                           dest_dir: Directory[FL2], dest_part: str):
        self._check_pre_copy(source_dir, source_part, dest_dir, dest_part)
        
        source_file = source_dir.files[source_part]
        if source_file.is_alive(timestamp) != True:
            raise RuntimeError(f"Source file {source_part} does not exist.")

    def file_copy_at(self, timestamp: float, file_from: str, file_to: str):
        source_dir, source_part = self._resolve_path(file_from)
        dest_dir, dest_part = self._resolve_path(file_to)

        self._check_pre_copy_l3(timestamp=timestamp, 
                                source_dir=source_dir,
                                source_part=file_from,
                                dest_dir=dest_dir,
                                dest_part=file_to)

        source_file = source_dir.files[source_part]
        factory_args = {"file_name": dest_part, "size": source_file.size, "timestamp": timestamp}
        copied_file = self._file_factory(**factory_args)
        self._store_file(dest_dir, dest_part, copied_file)
        print("L3 Copied {file_from} to {file_to} at timestamp {timestamp}")

    def file_search_at(self, timestamp: float, prefix: str) -> list[FL2]:
        """
        - Results should only include files that are still "alive"
        """
        current_dir, file_part = self._resolve_path(prefix)

        matching_files: list[tuple[int, str]] = []
        for file_name, file in current_dir.files.items():
            if file_name.startswith(file_part) and file.is_alive(timestamp):
                heapq.heappush(matching_files, (-file.size, file_name))

        # Get the top 10 results
        top_10: list[tuple[str, int]] = []
        count = 0
        while matching_files and count < 10:
            neg_size, name = heapq.heappop(matching_files)
            # heappop returns the smallest item
            top_10.append((name, -neg_size))
            count += 1
        
        return top_10

if MAIN:
    factory = FileFactoryL2()
    service = FileHostingServiceL3[FileL2, FileFactoryL2](file_factory=factory)
    now = 1
    
    service.file_upload_at(timestamp=1, file_name="file-1.zip", file_size=2, ttl=1)
    assert service.file_get_at(timestamp=1, file_name="file-1.zip") == 2
    assert service.file_get_at(timestamp=2, file_name="file-1.zip") == None
    
    service.file_copy_at(timestamp=1, file_from="file-1.zip", file_to="file-1-copy.zip")
    assert service.file_get_at(timestamp=1, file_name="file-1-copy.zip") == 2
    assert service.file_get_at(timestamp=2, file_name="file-1-copy.zip") == 2

    service.file_upload_at(timestamp=2, file_name="file-2.txt", file_size=1, ttl=1)
    service.file_upload_at(timestamp=2, file_name="file-3.txt", file_size=3, ttl=1)
    service.file_upload_at(timestamp=2, file_name="file-4.txt", file_size=4, ttl=1)
    service.file_upload_at(timestamp=2, file_name="file-5.txt", file_size=5, ttl=1)
    service.file_upload_at(timestamp=2, file_name="file-6.txt", file_size=6, ttl=1)
    service.file_upload_at(timestamp=2, file_name="file-7.txt", file_size=7, ttl=1)
    service.file_upload_at(timestamp=2, file_name="file-8.txt", file_size=8, ttl=1)
    service.file_upload_at(timestamp=2, file_name="file-9.txt", file_size=9, ttl=1)
    service.file_upload_at(timestamp=2, file_name="file-10.txt", file_size=10, ttl=1)
    service.file_upload_at(timestamp=2, file_name="file-11.txt", file_size=11, ttl=1)
    print(f"Search for 'file': {service.file_search_at(timestamp=2, prefix='file')}")
    print(f"Search for 'filt': {service.file_search_at(timestamp=2, prefix='filt')}")

# %% Level 4

class FileHostingServiceL4(FileHostingServiceL3[FL2, FF2]):
    def __init__(self, file_factory):
        super().__init__(file_factory=file_factory)
    
    def rollback(self, timestamp: float):
        """
        - Rollback the state of the file storage to the state specified in the timestamp
        - All ttls should be recalculated accordingly
        """
        def recursive_rollback(current_dir: Directory[FL2]):
            for file_name, file in list(current_dir.files.items()):
                if file.timestamp > timestamp:
                    current_dir.files.pop(file_name)
                    continue
            
            for subdir_name, subdir in current_dir.subdirectories.items():
                recursive_rollback(subdir)
        
        recursive_rollback(self.root)
        print(f"Rolled back to timestamp {timestamp}.")

if MAIN:
    factory = FileFactoryL2()
    service = FileHostingServiceL4[FileL2, FileFactoryL2](file_factory=factory)
    now = 1
    service.file_upload_at(timestamp=1, file_name="dir-a/file-1.txt", file_size=2, ttl=1)
    service.file_upload_at(timestamp=2, file_name="dir-b/file-2.txt", file_size=1, ttl=1)
    service.rollback(timestamp=1)
    assert service.file_get_at(timestamp=1, file_name="dir-a/file-1.txt") == 2
    assert service.file_get_at(timestamp=1, file_name="dir-b/file-2.txt") == None

# %%
