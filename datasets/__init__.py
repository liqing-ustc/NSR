from .scan import SCAN
from .pcfg import PCFG
from .hint import HINT

def get_dataset(name):
	if name == 'scan':
		return SCAN
	if name == 'pcfg':
		return PCFG
	if name == 'hint':
		return HINT
	
	assert False, f'Unknown dataset: {name}'
	
