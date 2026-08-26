import os

from .._pckg_exceptions._pckg_errors import *
from .._pckg_exceptions._pckg_warnings import *

class FileTree:

    tree_sets = {}

    ALLOWED_EXTENSIONS = ('.csv', '.xlsx', '.xls', '.xml')

    def __init__(self):
        self.tree = {}
        self.root_name = '.'

    def _is_data_file(self, name):
        return name.lower().endswith(self.ALLOWED_EXTENSIONS)

    def _build(self, root_path):
        tree = {}
        for entry in os.scandir(root_path):
            if entry.is_dir():
                tree[entry.name] = self._build(entry.path)
            elif self._is_data_file(entry.name):
                tree[entry.name] = str(os.path.join(root_path, entry.name))
        return tree

    def _guard(self, tree=None, path='.'):
        """
        Each folder can only contain either subfolders or data files,
        but not both. Raises ValueError on the first violation.
        """
        if tree is None:
            tree = self.tree

        folders = {name: sub for name, sub in tree.items() if isinstance(sub, dict)}
        files = {name for name, sub in tree.items() if not isinstance(sub, dict)}

        if folders and files:
            raise FileFinderError(
                f"Invalid structure at '{path}': folder contains both "
                f"subfolders and data files.\n"
                f"  Subfolders: {sorted(folders)}\n"
                f"  Data files: {sorted(files)}"
            )

        for name, subtree in folders.items():
            self._guard(subtree, path=f"{path}/{name}")


    def make_tree(self, root_path):
        """
        Builds the file tree starting from the given root path (main directory).

        Returns:
            FileTree: An instance of the FileTree with the constructed tree.

        Result methods:
        ----
        >>> result.show()  # Displays the tree structure
        >>> result.get('dict')  # Returns the tree as a dictionary
        """

        self.root_name = os.path.basename(os.path.abspath(root_path))
        self.tree = self._build(root_path)
        self._guard()
        return self
    

    def show(self, tree=None, root_name=None, prefix=''):
        """
        Displays the file tree structure in a readable scheme format.

        Parameters:
            tree (dict, optional): The tree structure to display. Defaults to None (uses self.tree).
            root_name (str, optional): The name of the root directory. Defaults to None (uses self.root_name).
            prefix (str, optional): The prefix for formatting the output. Defaults to ''.
        """

        if tree is None:
            tree = self.tree

        if prefix == '':
            print(root_name if root_name is not None else self.root_name)
        
        if not isinstance(tree, str):
            entries = list(tree.items())
        else:
            entries = []

        for index, (name, subtree) in enumerate(entries):
            is_last = index == len(entries) - 1
            connector = '└── ' if is_last else '├── '
            print(prefix + connector + name)
            if subtree is not None:
                extension = '    ' if is_last else '│   '
                self.show(subtree, root_name=name, prefix=prefix + extension)

    def get(self, type: str = 'dict'):
        """
        Returns the file tree in the specified format ('dict' or 'list') 
        for further processing - loading data from the given directory.

        Parameters:
            type (str): The desired format of the tree. 
                        'dict' returns a nested dictionary structure.
                        'list' returns a list representation of the tree.
        """
        current = self.tree
        if type == 'dict':
            return current
        elif type == 'list':
            return self._tree_to_list(current)
        else:
            raise ValueError("Invalid type. Use 'dict' or 'list'.")
        

    def _tree_to_list(self, tree):
        result = []
        for name, subtree in tree.items():
            if isinstance(subtree, dict):
                result.append(self._tree_to_list(subtree))
            else:
                result.append(subtree)
        return result


make_tree = FileTree().make_tree
