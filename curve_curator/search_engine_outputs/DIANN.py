import numpy as np


class DiannMap:
    def __init__(self, version):
        version = version.split('.')
        self.version_major = version[0]
        self.version_minor = version[1]

    @staticmethod
    def rename_general_columns(cols):
        return cols.map(lambda c: DiannMap._col_map.get(c, c))

    @staticmethod
    def rename_intensity_columns(cols):
        return cols.str.replace(r'Intensity.', 'Raw ', regex=True)

    _col_map = {
        'Genes': 'Genes',
        'Protein.Ids': 'Proteins',
    }
