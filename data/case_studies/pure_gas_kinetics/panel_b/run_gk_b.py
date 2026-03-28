# NOTE: total_initial_concentration must be defined for architectural consistency 
# when particle and CNT dynamics are disabled. Setting carb_struct_enabled=False 
# bypasses particle-gas interactions and reactor dynamics.


from pathlib import Path
from carbonx import GasReactor
from carbonx.modules.simulation_setup_loader import build_kwargs

SETUP_FILE = Path("simulation_setup.txt")
model = GasReactor(
    **build_kwargs(
        SETUP_FILE,
        total_pressure=8000,
        surface_kinetics_solver_activated=True,
        carb_struct_enabled=False,
        catalsyt_element="Fe",
        intnum= 27,
        bin_spacing=1.9,
        length_step = 'flex_loose',
        temperature_history="custom",
        total_initial_concentration=1e8,
        __xqtot=3.666421e-06,
        reactor_length=0.044,
        xdtube=.022,
        gas_initial_composition={'C2H4' :   0.994,'C2H2' :   1e-15,'CH4' :  0.0019,   "C2H6" :   0.004},
        dp_initial_premade=5.6e-9,
        kinetics_mechanism_type='Caltech')) 

_, solutions = model.solve()




