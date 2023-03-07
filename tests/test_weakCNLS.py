# import packages
from pystoned2 import weakCNLS
from pystoned2.constant import FUN_PROD, OPT_LOCAL,CET_ADDI
from pystoned2 import dataset as dataset

# import the GHG emissions data
data = dataset.load_GHG_abatement_cost()

# define and solve the CNLS-DDF model

def test_weakCNLS():
    model = weakCNLS.weakCNLS(y=data.y, x=data.x, b=data.b, cet=CET_ADDI, fun=FUN_PROD)
    model.optimize( solver="mosek")

    # display the estimates (alpha, beta, gamma, delta, and residual)
    model.display_alpha()
    model.display_beta()
    model.display_delta()
    model.display_residual()