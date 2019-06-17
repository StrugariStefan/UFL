# -*- mode: python -*-

block_cipher = None

import sys
sys.setrecursionlimit(5000)

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
datas = collect_data_files("skimage.io._plugins")
hiddenimports = collect_submodules('skimage.io._plugins') + ['pywt',
'pywt._extensions._cwt'] + ['cython', 'sklearn', 'sklearn.ensemble', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree._utils']

a = Analysis(['ufl.py'],
             pathex=['C:\\Users\\Stefan\\Downloads\\InfoFacultate\\ML\\licenta\\object_recognition_cli'],
             binaries=[],
             datas=datas,
             hiddenimports=hiddenimports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='ufl',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='ufl')
