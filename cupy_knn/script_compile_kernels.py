import cupy as cp
import numpy as np
import pathlib
from typing import Optional
from cuda_util import select_block_grid_sizes, get_cuda_include_path

_cuda_include = get_cuda_include_path()

_file_path = pathlib.Path(__file__).parent

with open(_file_path / "cuda/lbvh_kernels.cu", 'r') as f:
    _lbvh_src = f.read()

_compile_flags = ('--std=c++14', f' -I{pathlib.Path(__file__).parent / "cuda"}',
                  f'-I{_cuda_include}', '--use_fast_math', '--extra-device-vectorization')

_construct_tree_kernels = cp.RawModule(code=_lbvh_src,
                                       options=_compile_flags,
                                       name_expressions=('compute_morton_kernel',
                                                         'compute_morton_points_kernel',
                                                         'initialize_tree_kernel',
                                                         'construct_tree_kernel',
                                                         'optimize_tree_kernel',
                                                         'compute_free_indices_kernel',
                                                         'compact_tree_kernel'))

_compute_morton_kernel_float = _construct_tree_kernels.get_function('compute_morton_kernel')