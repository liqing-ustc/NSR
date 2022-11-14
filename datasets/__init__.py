from .scan import SCAN
from .pcfg import PCFG
from .hint import HINT
from .mt import MT
from .helper import MISSING_VALUE, EMPTY_VALUE

def get_dataset(name):
	if name == 'scan':
		return SCAN
	if name == 'pcfg':
		return PCFG
	if name == 'hint':
		return HINT
	if name == 'mt':
		return MT
		
	assert False, f'Unknown dataset: {name}'
	
