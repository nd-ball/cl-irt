"""
Models will subclass pytorch models.
Require training functions just like pytorch models
Is there anything special I need here? 
Maybe not.
So I might not need a CLModel class. 
We'll see about the other implementations. 
Nevermind, definitely do. 
They need to account for difficulties somehow. 
But is that a data thing or a model thing? 
We'll see. 
"""

import abc 
from typing import Dict, Any


class AbstractModel(abc.ABC):
    def __init__(self, *args):
        pass    
    
    @abc.abstractmethod 
    def forward(self, x):
        pass 



