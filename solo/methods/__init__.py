# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


from solo.methods.base import BaseMethod
from solo.methods.mocov2plus import MoCoV2Plus

# dual temperature method
from solo.methods.simco_dual_temperature import SimCo_DualTemperature
from solo.methods.simmoco_dual_temperature import SimMoCo_DualTemperature
from solo.methods.mocov2 import MoCoV2

METHODS = {
    # base classes
    "base": BaseMethod,
    # methods
    "mocov2plus": MoCoV2Plus,
    
    "simco_dual_temperature": SimCo_DualTemperature,
    "simmoco_dual_temperature": SimMoCo_DualTemperature,
    "mocov2": MoCoV2,

}

__all__ = [
    "BaseMethod",
    "MoCoV2Plus",
    "SimCo_DualTemperature",
    "SimMoCo_DualTemperature",
    "MoCoV2",
]

try:
    from solo.methods import dali  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("dali")
