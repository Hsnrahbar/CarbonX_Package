'''
Note: Surface_Kinetics_General_UDF is set when dual-dissociation model is used.
For more information, please  refer to CarbonX manual.
'''

from pathlib import Path
from carbonx import MappingWrapper
from carbonx . modules . simulation_setup_loader import build_kwargs

SETUP_FILE = Path(__file__).with_name("simulation_setup.txt")

model = MappingWrapper (map_requested="P&T",
                          grid_total =1025,
                          P_range_min =0.001 ,
                          P_range_max =0.01 ,
                          P_iso =0.01 ,
                          T_range_min =800 ,
                          T_range_max =1200 ,
                          T_iso =873 ,
                          xdp_range_min =10e-9 ,
                          xdp_range_max =50e-9 ,
                          xdp_const =15e-9 ,
                          L_reactor_range_min =0.1 ,
                          L_reactor_range_max =1 ,
                          L_reactor =0.6 ,
                          xN_range_min =1e17 ,
                          xN_range_max =10e17 ,
                          xN_const =1e11 ,
                          scale_min =10e-8 ,
                          scale_max =7e-6 ,
                          ml_method = " mean " ,
                          ml_lambda_ =0.001 ,
                          ml_iterations =10000 ,
                          ml_alpha =1.5 ,
                          surface_kinetics_type = "Surface_Kinetics_General_UDF" , # "multilayered_model"
                          ml_post_cond = True) 
 
_, solutions = model.run()


results = model.run_parametric_study()
model.parametricstudyvisualizer()


