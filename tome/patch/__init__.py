# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from .swag import apply_patch as swag
from .timm import apply_patch as timm
from .mae  import apply_patch as mae

__all__ = ["timm", "swag", "mae"]
