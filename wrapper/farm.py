import os
import numpy as np
from openfast_toolbox.io import FASTInputFile
from .openfast_base import OpenFASTFile


class Farm(FASTInputFile):

    def __init__(self, base_filepath: str, filename: str | None = None):
        """
        Wrapper/Facade para gestionar archivos de granjas OpenFAST (.fstf).
        Sigue la estructura estándar evitando conflictos con las propiedades base.
        """
        if not base_filepath.endswith(".fstf"):
            raise ValueError("Base filepath must be a farm file (.fstf)")

        self.base_filepath = base_filepath

        # Usamos un atributo privado para evitar colisionar con el 'filename' de la clase padre
        self._user_filename = filename

        # Inicializamos la clase base leyendo la plantilla original
        super().__init__(base_filepath)

        # Si se especificó un filename desde el inicio, calculamos la ruta destino final
        if filename is not None:
            filename_clean = os.path.splitext(filename)[0]
            base_dir = os.path.dirname(self.base_filepath)
            self.filepath = os.path.join(base_dir, f"{filename_clean}.fstf")
        else:
            self.filepath = base_filepath

    @property
    def filename_override(self) -> str | None:
        """Retorna el nombre de archivo personalizado asignado por el usuario."""
        return self._user_filename

    def toOFF(self, filename: str = None) -> OpenFASTFile:
        self.toFile()
        return OpenFASTFile(self.filepath, filename)

    def addWT(
        self,
        pos: np.ndarray,
        turbine: str | OpenFASTFile,
        high_res_origin: np.ndarray = np.array([0.0, 0.0, 0.0]),
        high_res_grid: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ):
        """
        Adds a wind turbine, converting all numeric values to string format
        and quoting the turbine filepath to perfectly match the OpenFAST template format.
        """
        g = list(self["WindTurbines"])

        # 1. Obtener y calcular la ruta relativa de la turbina
        turbine_file = (
            turbine.filepath if isinstance(turbine, OpenFASTFile) else turbine
        )
        template_dir = os.path.dirname(self.base_filepath)
        turbine_relative_file = os.path.relpath(turbine_file, start=template_dir)

        # 2. Asegurar que el nombre del archivo lleve comillas dobles ej: '"WT1.fst"'
        if not turbine_relative_file.startswith('"'):
            turbine_relative_file = f'"{turbine_relative_file}"'

        # 3. Empaquetar todo convirtiendo de flotantes a cadenas de texto
        # Usamos f-strings para formatear con precisión decimal limpia (ej: '0.0')
        turbine_array = np.array(
            [
                f"{pos[0]:.1f}",
                f"{pos[1]:.1f}",
                f"{pos[2]:.1f}",
                turbine_relative_file,
                f"{high_res_origin[0]:.1f}",
                f"{high_res_origin[1]:.1f}",
                f"{high_res_origin[2]:.1f}",
                f"{high_res_grid[0]:.1f}",
                f"{high_res_grid[1]:.1f}",
                f"{high_res_grid[2]:.1f}",
            ],
            dtype=object,
        )

        g.append(turbine_array)
        self["WindTurbines"] = g

    def toFile(self, filename: str | None = None):
        """
        Guarda el archivo en el disco.
        Si se pasa un filename, se guarda relativo al directorio de base_filepath.
        Si no se pasa nada, guarda en la ruta precalculada (self.filepath).
        """
        if filename is None:
            save_path = self.filepath
        else:
            base_dir = os.path.dirname(self.base_filepath)
            filename_clean = os.path.splitext(filename)[0]
            save_path = os.path.join(base_dir, f"{filename_clean}.fstf")

            # Actualizamos nuestros atributos de control interno
            self._user_filename = filename
            self.filepath = save_path

        # Invocamos la escritura nativa de openfast_toolbox
        self.write(save_path)


class AeroAcousticObservers:
    def __init__(self, turbine_pos: np.ndarray = None):
        """
        Initializes the observer manager.
        :param turbine_pos: The 3D coordinate array of the turbine [X, Y, Z].
        """
        self.observers = []

        # Safely assign the default array if no argument is passed
        if turbine_pos is None:
            self.turbine_pos = np.array([0, 0, 0])
        else:
            self.turbine_pos = np.asarray(turbine_pos)

    def addObservers(self, obs_pos: np.ndarray, is_absolute: bool = False):
        """
        Adds one or more observer positions.
        Accepts a single 3D coordinate (e.g., [x, y, z]) or a batch of them (e.g., [[x1, y1, z1], [x2, y2, z2]]).
        """
        # Convert input to a NumPy array
        obs_array = np.asarray(obs_pos)

        # Check if the input is 1D (single coordinate) or 2D (multiple coordinates)
        if obs_array.ndim == 1:
            # Wrap in a list so we can loop over it uniformly
            coords_to_process = [obs_array]
        elif obs_array.ndim == 2:
            coords_to_process = obs_array
        else:
            raise ValueError(
                "obs_pos must be a 1D coordinate or a 2D array of coordinates."
            )

        # Process and append each coordinate
        for coord in coords_to_process:
            if is_absolute:
                if self.turbine_pos is None:
                    raise ValueError(
                        "turbine_pos must be provided during class initialization if is_absolute is True."
                    )
                relative_pos = coord - self.turbine_pos
                self.observers.append(relative_pos)
            else:
                self.observers.append(coord)

    def toFile(self, filename: str):
        """
        Exports using NumPy's file saving utility or formatting.
        """
        num_obs = len(self.observers)

        with open(filename, "w") as f:
            f.write(f"{num_obs} NrObsLoc - Total Number of observer locations\n")
            f.write(
                "X Observer location in tower-base coordinate X horizontal (m), Y Observer location in tower-base coordinate Y Lateral (m), Z Observer location in tower-base coordinate Z Vertical (m)\n"
            )

            # Write out each position vector neatly
            for obs in self.observers:
                f.write(f"{obs[0]:>9.2f} {obs[1]:>11.2f} {obs[2]:>6.1f}\n")

    def linkToAA(self, AA_file):
        """
        Links the saved observer locations to the AeroAcoustic file object.
        """
        base, _ = os.path.splitext(AA_file.filepath)
        filename = f"{base}_obs.dat"

        # Save the observers to the newly constructed filename
        self.toFile(filename)

        if hasattr(AA_file, "ObserverLocations"):
            AA_file.ObserverLocations.link(filename)
        else:
            raise AttributeError(
                "The provided AA_file does not have an 'ObserverLocations' attribute."
            )
