# -*- mode: python ; coding: utf-8 -*-

import glob
import os
import cv2

def get_opencv_binaries():
    base_dir = os.path.dirname(cv2.__file__)
    binaries = glob.glob(os.path.join(base_dir, '*.so*'))
    binaries += glob.glob(os.path.join(base_dir, 'cv2/*.so*'))

    return [(binary, '.') for binary in binaries]

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=get_opencv_binaries(),
    datas=[],
    hiddenimports=['cv2', 'numpy', 'matplotlib', 'scipy.ndimage', 'numpy.linalg', 'matplotlib.animation'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)