import os
from .utils import jprint


class OpenFASTValue(str):
    """
    A specialized string type for OpenFAST values.
    Behaves exactly like a standard string, but carries a .__doc__ attribute
    and enables contextual .open() and .link() side-effect actions.
    """

    def __init__(self, value):
        super().__init__()
        self.__doc__ = ""
        self._parent_obj = None
        self._key_name = None

    def _set_context(self, parent_obj, key_name, doc_str):
        self._parent_obj = parent_obj
        self._key_name = key_name
        self.__doc__ = doc_str

    def open(self, autoname=True):
        if not self._parent_obj:
            raise ValueError("File link missing context configuration.")

        if self.lower() in ["unused", "default", ""] or not self:
            raise ValueError(
                f"Property '{self._key_name}' does not point to an active file link."
            )

        # Import inside the method to prevent circular reference errors
        from .openfast_base import OpenFASTFile

        parent_dir = os.path.dirname(self._parent_obj.filepath)
        resolved_path = os.path.normpath(os.path.join(parent_dir, self))
        jprint(f"[+] Opening linked file: '{self}' -> '{resolved_path}'")

        if autoname:
            # 1. Get the parent filename (e.g., 'WT2.fst')
            parent_leaf = os.path.basename(self._parent_obj.filepath)
            # 2. Strip its extension (e.g., 'WT2')
            parent_base, _ = os.path.splitext(parent_leaf)
            # 3. Combine with key name (e.g., 'WT2.AeroFile')
            constructed_filename = f"{parent_base}.{self._key_name}"

            return OpenFASTFile(resolved_path, filename=constructed_filename)

        return OpenFASTFile(resolved_path)

    def link(self, target):
        if not self._parent_obj:
            raise ValueError("File link missing context configuration.")

        from .openfast_base import OpenFASTFile

        if isinstance(target, OpenFASTFile):
            abs_target = os.path.abspath(target.filepath)
        else:
            abs_target = os.path.abspath(str(target))

        parent_dir = os.path.dirname(self._parent_obj.filepath)
        relative_path = os.path.relpath(abs_target, parent_dir)
        normalized_link = relative_path.replace("\\", "/")

        self._parent_obj[self._key_name] = normalized_link
        jprint(f"[+] Updated parent variable '{self._key_name}' -> '{normalized_link}'")

    def ol(self, autoName=True):
        """
        Opens and links a file

        Args:
          autoName: hi
        """
        f = self.open()
        self.link(f)
        return f
