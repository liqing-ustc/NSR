import torch
import numpy as np
np.set_printoptions(precision=2, suppress=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from data.scan import MISSING_VALUE, EMPTY_VALUE, SCAN

SYMBOLS = SCAN.i2w
SYM2ID = lambda x: SCAN.w2i.get(x)
ID2SYM = lambda x: SCAN.i2w[x]
SYM2PROG = SCAN.sym2prog
SYM2ARITY = SCAN.sym2arity