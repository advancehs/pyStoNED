# import packages
from pystoned import CNLS
from pystoned.constant import CET_ADDI, FUN_COST, RTS_CRS, RED_MOM
from pystoned.dataset import load_Finnish_electricity_firm

# import all data (including the contextual varibale)
data = load_Finnish_electricity_firm(x_select=['Energy', 'Length', 'Customers'],
                                     y_select=['TOTEX'],
                                     z_select=['PerUndGr'])

# define and solve the CNLS-Z model
model = CNLS.CNLS(y=data.y, x=data.x, z=data.z, cet = CET_ADDI, fun = FUN_COST, rts = RTS_CRS)
model.optimize( solver="mosek")

# display the coefficient of contextual variable
model.get_frontier()