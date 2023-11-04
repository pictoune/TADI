from horn import optimize_horn_schunck
from utils import display_results

smallest_alpha = 0.1
biggest_alpha = 200
step = 5

dataset_name = "yosemite"
involved_parameter = "alpha"

min_error,best_flow,best_alpha,errors = optimize_horn_schunck(dataset_name,smallest_alpha,biggest_alpha,step,nb_iter=10)

display_results(smallest_alpha,biggest_alpha,step,best_flow,errors,involved_parameter,best_alpha,dataset_name)