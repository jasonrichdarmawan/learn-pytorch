# I failed because I overcomplicated the code.
# I should have stored the files in a `files: list[File]`.
# Want to find a file with specific prefix and suffix?
# Just use a list comprehension.
# But, because I created the `Dictionary` and `File` classes, I need to use recursive function
# I am afraid my mistake already cost me the opportunity.

# %%

from typing import Dict

MAIN = "__main__" == __name__

# %%

class File:
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size

    def __repr__(self):
        return f"File(name={self.name},size={self.size})"
    
if MAIN:
    file = File(name="A", size=1)
    print(file)

# %%

class Directory:
    def __init__(self, name: str):
        self.name = name
        self.files: Dict[str, File] = {}
        self.subdirectories: Dict[str, Directory] = {}

    def __repr__(self):
        return (f"Directory("
                f"name={self.name},\n"
                f"files={self.files},\n"
                f"subdirectories={self.subdirectories})")

if MAIN:
    directory = Directory(name="A")
    directory.files["B"] = File(name="B", size=1)
    directory.subdirectories["C"] = Directory(name="C")
    directory.subdirectories["C"].files["D"] = File(name="D", size=2)
    print(directory)

# %%

class FileSystem:
    def __init__(self):
        self.root = Directory(name="/")
        self.current_directory = self.root

    def __repr__(self):
        return f"FileSystem(root={self.root})"
    
if MAIN:
    filesystem = FileSystem()
    filesystem.root.files["A"] = File(name="A", size=1)
    filesystem.root.subdirectories["B"] = Directory(name="B")
    filesystem.root.subdirectories["B"].files["C"] = File(name="C", size=2)
    print(filesystem)

# %%
