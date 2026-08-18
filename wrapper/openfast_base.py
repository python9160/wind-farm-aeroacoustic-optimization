import os
import re

from .utils import jprint
from .openfast_value import OpenFASTValue

class OpenFASTFile:
    """
    A class to represent and manipulate OpenFAST configuration files (.fst, .dat, etc.).
    """
    def __init__(self, base_filepath, filename = None):
        self.base_filepath = os.path.abspath(base_filepath)
        
        if filename:
            # Get the directory containing the base file
            base_dir = os.path.dirname(self.base_filepath)
            
            # Extract the extension from the base_filepath (e.g., '.fst' or '.dat')
            _, ext = os.path.splitext(self.base_filepath)
            
            # Combine them: directory + extensionless filename + original extension
            self.filepath = os.path.join(base_dir, f"{filename}{ext}")
        else:
            # Default fallback if no filename is passed
            self.filepath = self.base_filepath
            
        self.lines = []
        self.data_map = {}
        self.doc_map = {}
        self._load_file()

    def _load_file(self):
        with open(self.base_filepath, 'r') as f:
            self.lines = f.readlines()
        
        # Regex looks for: Value, Variable name, and Comment after '-'
        pattern = re.compile(r'^\s*(.*?)\s+([a-zA-Z0-9_\(\),]+)\s+-\s*(.*)$')
        
        for idx, line in enumerate(self.lines):
            stripped = line.strip()
            
            # GUARD CLAUSE: Skip header separators and file termination lines
            if (stripped.startswith('-') or 
                stripped.startswith('=') or 
                stripped.upper().startswith('END')):
                continue
                
            match = pattern.match(line)
            if match:
                var_name = match.group(2).strip()
                doc_text = match.group(3).strip()
                
                self.data_map[var_name] = idx
                self.doc_map[var_name] = doc_text
                
                # Expose clean names as properties for autocomplete fields
                if "(" not in var_name and not hasattr(self, var_name):
                    setattr(self, var_name, None)

    def _ipython_key_completions_(self):
        return list(self.data_map.keys())

    def __getitem__(self, key):
        if key not in self.data_map:
            raise KeyError(f"Variable '{key}' not found in this configuration file.")
        
        line_idx = self.data_map[key]
        line = self.lines[line_idx]
        
        match = re.match(r'^\s*(.*?)\s+' + re.escape(key), line)
        if match:
            val_str = match.group(1).strip()
            if val_str.startswith('"') and val_str.endswith('"'):
                val_str = val_str.strip('"')
                
            val_obj = OpenFASTValue(val_str)
            val_obj._set_context(self, key, self.doc_map.get(key, ""))
            return val_obj
        return None

    def __setitem__(self, key, value):
        if key not in self.data_map:
            raise KeyError(f"Variable '{key}' cannot be modified because it doesn't exist.")
        
        line_idx = self.data_map[key]
        line = self.lines[line_idx]
        
        pattern = re.compile(r'^(\s*.*?)\s+(' + re.escape(key) + r'\s+-.*)$')
        match = pattern.match(line)
        
        if match:
            if isinstance(value, str) and not value.startswith('"') and ('/' in value or '\\' in value or value in ["unused", "FATAL", "default", "G0"]):
                formatted_val = f'"{value}"'
            else:
                formatted_val = str(value)
                
            new_line = f"{formatted_val:>14}   {match.group(2)}\n"
            self.lines[line_idx] = new_line

    def __getattr__(self, name):
        if name in self.data_map:
            return self.__getitem__(name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in ['filepath', 'base_filepath', 'lines', 'data_map', 'doc_map'] or not hasattr(self, 'data_map'):
            super().__setattr__(name, value)
        elif name in self.data_map:
            self.__setitem__(name, value)
        else:
            super().__setattr__(name, value)

    def __dir__(self):
        return sorted(super().__dir__() + list(self.data_map.keys()))

    def toFile(self, filename=None):
        if filename is None:
            save_path = self.filepath
        else:
            base_dir = os.path.dirname(self.base_filepath)
            if not filename.endswith(('.fst', '.dat', '.txt', '.inp')):
                ext = os.path.splitext(self.base_filepath)[1]
                filename = f"{filename}{ext}"
            save_path = os.path.join(base_dir, filename)
            
        with open(save_path, 'w') as f:
            f.writelines(self.lines)
        jprint(f"File successfully saved to: {save_path}")
        return save_path