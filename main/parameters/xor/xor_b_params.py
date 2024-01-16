from typing import List, Optional

from parameters.xor.xor_parameters import XorParams


class XorBParams(XorParams):
    start_dimensions: List[int] = [6, 5, 4, 3, 2]
    num_layers: int = 4
    task_version_name: Optional[str] = 'xor_b'
