
from argparse import Namespace

import tsp
import os.path as osp
root_path = osp.dirname(osp.dirname(tsp.__file__))
output_path = osp.join(root_path, "approx_global_optima_datasets")


SUPER_CONFIG = dict(
    dscaling_10n_2swap = Namespace(
        mlflow_logging_signature = f"APPROX_global_optima/dim_scaling_10n_2swap",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
        output_dataset_path = output_path,
        
        dscales = list(range(2, 13)) + [15, 20, 30, 40, 50, 100],
        nscales = [10],  # problem size
        
        num_problems = 4_000,  # per dim scale
        optima_per_problem = 100,  # local optima found per problem starting from distinct random solutions
        search_algo = "2swap",  # supports only '2opt' and '2swap' currently

        parallel_jobs = 32,  # number of parallel jobs
        seed = None,  # seed random tour starts for local search, or use random seed if set to None
    ),

    dscaling_20n_2swap = Namespace(
        mlflow_logging_signature = f"APPROX_global_optima/dim_scaling_20n_2swap",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
        output_dataset_path = output_path,
        
        dscales = list(range(2, 13)) + [15, 20, 30, 40, 50, 100],
        nscales = [20],  # problem size
        
        num_problems = 4_000,  # per dim scale
        optima_per_problem = 100,  # local optima found per problem starting from distinct random solutions
        search_algo = "2swap",  # supports only '2opt' and '2swap' currently

        parallel_jobs = 32,  # number of parallel jobs
        seed = None,  # seed random tour starts for local search, or use random seed if set to None
    ),

    nscaling_2opt = Namespace(
        mlflow_logging_signature = f"APPROX_global_optima/node_scaling",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
        output_dataset_path = output_path,
        
        dscales = [2],
        nscales = [45, 50],  # problem size
        
        num_problems = 2_000,  # per dim scale
        optima_per_problem = 100,  # local optima found per problem starting from distinct random solutions
        search_algo = "2opt",  # supports only '2opt' and '2swap' currently

        parallel_jobs = 32,  # number of parallel jobs
        seed = None,  # seed random tour starts for local search, or use random seed if set to None
    ),
    
    nscaling_2swap = Namespace(
        mlflow_logging_signature = f"APPROX_global_optima/node_scaling_2swap",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
        output_dataset_path = output_path,
        
        dscales = [2],
        nscales = list(range(5, 51, 5)),  # problem size
        
        num_problems = 1_000,  # per dim scale
        optima_per_problem = 100,  # local optima found per problem starting from distinct random solutions
        search_algo = "2swap",  # supports only '2opt' and '2swap' currently

        parallel_jobs = 32,  # number of parallel jobs
        seed = None,  # seed random tour starts for local search, or use random seed if set to None
    ),

    ##########################################################
    
    bigk_nscaling_2opt = Namespace(
        mlflow_logging_signature = f"bigK_local_optima/node_scaling_2opt",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
        output_dataset_path = output_path,
        
        dscales = [2],
        nscales = list(range(5, 51, 5)),  # problem size
        
        num_problems = 1,  # per scale
        optima_per_problem = 100_000,  # local optima found per problem starting from distinct random solutions
        search_algo = "2opt",  # supports only '2opt' and '2swap' currently

        parallel_jobs = 32,  # number of parallel jobs
        seed = None,  # seed random tour starts for local search, or use random seed if set to None
    ),

    bigk_dscaling_10n_2opt = Namespace(
        mlflow_logging_signature = f"bigK_local_optima/dim_scaling_10n_2opt",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
        output_dataset_path = output_path,
        
        dscales = list(range(2, 13)) + [15, 20, 30, 40, 50, 100],
        nscales = [10],  # problem size
        
        num_problems = 64,  # NOTE dim scaling requires more precision, as the LO scales slower and is nonmonotonic
        optima_per_problem = 100_000,  # local optima found per problem starting from distinct random solutions
        search_algo = "2opt",  # supports only '2opt' and '2swap' currently

        parallel_jobs = 32,  # number of parallel jobs
        seed = None,  # seed random tour starts for local search, or use random seed if set to None
    ),

    bigk_dscaling_20n_2opt = Namespace(
        mlflow_logging_signature = f"bigK_local_optima/dim_scaling_20n_2opt",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
        output_dataset_path = output_path,
        
        dscales = list(range(2, 13)) + [15, 20, 30, 40, 50, 100],
        nscales = [20],  # problem size
        
        num_problems = 16,  # NOTE dim scaling requires more precision, as the LO scales slower and is nonmonotonic
        optima_per_problem = 100_000,  # local optima found per problem starting from distinct random solutions
        search_algo = "2opt",  # supports only '2opt' and '2swap' currently

        parallel_jobs = 32,  # number of parallel jobs
        seed = None,  # seed random tour starts for local search, or use random seed if set to None
    ),

    bigk_nscaling_2swap = Namespace(
        mlflow_logging_signature = f"bigK_local_optima/node_scaling_2swap",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
        output_dataset_path = output_path,
        
        dscales = [2],
        nscales = list(range(5, 51, 5)),  # problem size
        
        num_problems = 1,  # per scale
        optima_per_problem = 50_000,  # NOTE using less than 2opt due to compute time budget limitations
        search_algo = "2swap",  # supports only '2opt' and '2swap' currently

        parallel_jobs = 32,  # number of parallel jobs
        seed = None,  # seed random tour starts for local search, or use random seed if set to None
    ),


    bigk_dscaling_10n_2swap = Namespace(
        mlflow_logging_signature = f"bigK_local_optima/dim_scaling_10n_2swap",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
        output_dataset_path = output_path,
        
        dscales = list(range(2, 13)) + [15, 20, 30, 40, 50, 100],
        nscales = [10],  # problem size
        
        num_problems = 64,  # per dim scale
        optima_per_problem = 100_000,  # local optima found per problem starting from distinct random solutions
        search_algo = "2swap",  # supports only '2opt' and '2swap' currently

        parallel_jobs = 32,  # number of parallel jobs
        seed = None,  # seed random tour starts for local search, or use random seed if set to None
    ),

    bigk_dscaling_20n_2swap = Namespace(
        mlflow_logging_signature = f"bigK_local_optima/dim_scaling_20n_2swap",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
        output_dataset_path = output_path,
        
        dscales = list(range(2, 13)) + [15, 20, 30, 40, 50, 100],
        nscales = [20],  # problem size
        
        num_problems = 16,  # per dim scale
        optima_per_problem = 100_000,  # local optima found per problem starting from distinct random solutions
        search_algo = "2swap",  # supports only '2opt' and '2swap' currently

        parallel_jobs = 32,  # number of parallel jobs
        seed = None,  # seed random tour starts for local search, or use random seed if set to None
    ),
    
 
    

)