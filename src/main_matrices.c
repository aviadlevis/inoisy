/*
  Compile with:   make all
                  or make (insert model name) e.g. make disk
  
  Sample run:     mpirun -np 4 disk -n 32 -solver 0 -v 1 1
                  mpiexec -n 4 ./poisson -n 32 -solver 0 -v 1 1
  
  To see options: use option help, -help, or --help

  Index naming conventions:
  x0 = t, x1 = r, x2 = phi
  Given a stencil {i,j,k}, HYPRE has k as the slowest varying index. 
  Thus, the indices correspond to i = x2, j = x1, k = x0. 
*/

#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "hdf5_utils.h"
#include "param.h"
#include "model.h"

// TODO generalize dimension, set dimension in model

int main (int argc, char *argv[])
{
  int i, j, k;
  
  int myid, num_procs;
  
  int ni, nj, nk, pi, pj, pk, npi, npj, npk, seed, user_filename;
  double dx0, dx1, dx2, x0end, tol;
  int ilower[3], iupper[3];
  
  int solver_id, maxiter, verbose;
  int n_pre, n_post;
  
  clock_t start_t = clock();
  clock_t check_t;
  
  HYPRE_StructGrid    grid;
  HYPRE_StructStencil stencil;
  HYPRE_StructMatrix  A;
  HYPRE_StructVector  b;
  HYPRE_StructVector  x;
  HYPRE_StructSolver  solver;
  HYPRE_StructSolver  precond;
  
  int num_recursions; // number of recursions
  double final_res_norm;
  
  int output, timer, dump;

  char* dir_ptr;
  char* params_ptr;
  char* source_ptr;

  char filename[255];
  
  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  
  /* Set defaults */
  ni  = 32;                    /* nj and nk are set equal to ni later */
  npi = 1;
  npj = 1;
  npk = num_procs;             /* default processor grid is 1 x 1 x N */
  solver_id = 0;
  n_pre  = 1;
  n_post = 1;
  output = 1;                  /* output data by default */
  timer  = 0;
  x0end = 100.;
  dump   = 0;                  /* outputing intermediate steps if nrecur > 1 off by default */
  maxiter = 50;
  verbose = 2;
  tol = 1.0e-06;
  num_recursions = 1;
  char* default_dir = ".";     /* output in current directory by default */
  dir_ptr   = default_dir;
  params_ptr = NULL;
  source_ptr = NULL;
  int constrained = 0;
  int vinit = 0;
  /* Initiialize rng */
  const gsl_rng_type *T;
  gsl_rng *rstate;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  rstate = gsl_rng_alloc(T);
  seed = model_set_gsl_seed(gsl_rng_default_seed, myid);

  /* Parse command line */
  {
    user_filename = 0;
    int arg_index   = 0;
    int print_usage = 0;
    int check_pgrid = 0;

    while (arg_index < argc) {
      if ( strcmp(argv[arg_index], "-n") == 0 ) {
	arg_index++;
	ni = atoi(argv[arg_index++]);
      }
      else {
	arg_index++;
      }
    }
    nj = ni;
    nk = ni;
    
    arg_index = 0;
    while (arg_index < argc) {
      if ( strcmp(argv[arg_index], "-ni") == 0 ) {
	    arg_index++;
	    ni = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nj") == 0 ) {
	    arg_index++;
	    nj = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nk") == 0 ) {
	    arg_index++;
	    nk = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pgrid") == 0 ) {
	    arg_index++;
        /* Make sure there are 3 arguments after -pgrid */
        if (arg_index >= argc - 2) {
          check_pgrid = 1;
          break;
        }
        /* Check that pgrid agrees with assigned number of processors
           (Also checks for non-integer inputs) */
        npi = atoi(argv[arg_index++]);
        npj = atoi(argv[arg_index++]);
        npk = atoi(argv[arg_index++]);
        if ( num_procs != (npi * npj * npk) ) {
          check_pgrid = 1;
          break;
        }
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 ) {
	    arg_index++;
	    solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-v") == 0 ) {
	    arg_index++;
	    n_pre = atoi(argv[arg_index++]);
	    n_post = atoi(argv[arg_index++]);
      } // TODO check that there are two integers following -v
      else if ( strcmp(argv[arg_index], "-dryrun") == 0 ) {
	    arg_index++;
	    output = 0;
      }
      else if ( strcmp(argv[arg_index], "-ps") == 0 ||
		strcmp(argv[arg_index], "-sp") == 0 ) {
	    arg_index++;
	    params_ptr = argv[arg_index];
	    source_ptr = argv[arg_index++];
      }
      else if ( strcmp(argv[arg_index], "-p") == 0 ||
		strcmp(argv[arg_index], "-params") == 0 ) {
	    arg_index++;
	    params_ptr = argv[arg_index++];
      }
      else if ( strcmp(argv[arg_index], "-s") == 0 ||
		strcmp(argv[arg_index], "-source") == 0 ) {
	    arg_index++;
	    source_ptr = argv[arg_index++];
      }
      else if ( strcmp(argv[arg_index], "-o") == 0 ||
		strcmp(argv[arg_index], "-output") == 0 ) {
	    arg_index++;
	    dir_ptr = argv[arg_index++];
      } // TODO check that directory exists before solving rather than later
        // TODO remove trailing '/' if it exists
    else if ( strcmp(argv[arg_index], "-f") == 0 ||
		strcmp(argv[arg_index], "-filename") == 0 )  {
	    arg_index++;
	    sprintf(filename, "%s/%s", dir_ptr, argv[arg_index++]);
	    user_filename = 1;
      }
      else if ( strcmp(argv[arg_index], "-timer") == 0 ) {
	    arg_index++;
	    timer = 1;
      }
      else if ( strcmp(argv[arg_index], "-maxiter") == 0 ) {
        arg_index++;
        maxiter = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-verbose") == 0 ) {
        arg_index++;
        verbose = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dump") == 0 ) {
	    arg_index++;
	    dump = 1;
      }
      else if ( strcmp(argv[arg_index], "-constrained") == 0 ) {
        arg_index++;
	    constrained = 1;
      }
      else if ( strcmp(argv[arg_index], "-vinit") == 0 ) {
	    arg_index++;
	    vinit = 1;
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 ) {
	    arg_index++;
	    tol = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nrecur") == 0 ) {
	    arg_index++;
	    num_recursions = atoi(argv[arg_index++]);
	if (num_recursions < 1) {
	  print_usage = 1;
	  break;
	}
      }
      else if ( strcmp(argv[arg_index], "help")   == 0 ||
		strcmp(argv[arg_index], "-help")  == 0 ||
		strcmp(argv[arg_index], "--help") == 0 ) {
	    print_usage = 1;
	    break;
      }
    else if ( strcmp(argv[arg_index], "-seed") == 0 ||
		strcmp(argv[arg_index], "--seed") == 0 ) {
	    arg_index++;
	    seed = model_set_gsl_seed(atoi(argv[arg_index++]), myid);
      }
    else if ( strcmp(argv[arg_index], "-tend") == 0 ||
		strcmp(argv[arg_index], "--tend") == 0 ) {
	    arg_index++;
	    x0end = atof(argv[arg_index++]);
      }
      else {
	    arg_index++;
      }
    }
    
    if (print_usage) {
      if (myid == 0) {
	printf("\n");
	printf("Usage: %s [<options>]\n", argv[0]);
	printf("\n");
	printf("  -n <n>                 : General grid side length per processor (default: 32).\n");
	printf("  -ni <n> (or nj/nk)     : Grid side length for specific side per processor\n");
	printf("                           (default: 32). If both -n and -ni are specified, ni\n");
	printf("                           will overwrite n for that side only. \n");
	printf("  -pgrid <pi> <pj> <pk>  : Processor grid layout (default: 1 x 1 x num_procs).\n");
	printf("                           The product of pi * pj * pk must equal num_proc.\n");
	printf("  -solver <ID>           : solver ID\n");
	printf("                           0  - PCG with SMG precond (default)\n");
	printf("                           1  - SMG\n");
    printf("                           2  - Compute A*x matrix multiplication\n");
    printf("                           3  - Compute a gradient with respect to spatial angle\n");
    printf("                           4  - LOBPG eigenvectors/values\n");
    printf("                           5  - LOBPG eigenvectors/values with SMG preconditioner\n");
	printf("  -v <n_pre> <n_post>    : Number of pre and post relaxations (default: 1 1).\n");
	printf("  -dryrun                : Run solver w/o data output.\n");
	printf("  -params <file> (or -p) : Read in parameters from <file>.\n");
	printf("  -source <file> (or -s) : Read in source field from <file.\n");
	printf("                           Can be combined by using -ps or -sp\n");
	printf("  -output <dir> (or -o)  : Output data in directory <dir> (default: ./).\n");
	printf("  -filename (or -f)      : Output data filename. (default: <model_name_datetime.h5>).\n");
	printf("  -timer                 : Time each step on processor zero.\n");
	printf("  -dump                  : Output intermediate steps if nrecur > 1).\n");
	printf("  -nrecur                : Number of recursions to apply to source (default: 1).\n");
	printf("  -seed                  : Pass a user generated seed.\n");
	printf("  -tend                  : The end time for the simulation in terms of M. Default is 100. .\n");
    printf("  -maxiter               : Maximum number of solution iterations (default = 50).\n");
    printf("  -tol                   : Solver stop criteria (default = 1e-6).\n");
    printf("  -constrained           : Constrain the eigenvectors y'x=0.\n");
    printf("  -vinit                 : Initialize vector modes. \n");
    printf("  -verbose               : Level of verbosity (default = 2).\n");
	printf("\n");
	printf("Sample run:     mpirun -np 8 poisson -n 32 -nk 64 -pgrid 1 2 4 -solver 1\n");
	printf("                mpiexec -n 4 ./disk -n 128 -nj 32 -pgrid 2 2 1 -solver 0\n");
	printf("\n");
	printf("GSL_RNG_SEED=${RANDOM} mpiexec -n 4 ./general_xy -n 64 -nk 128 -pgrid 1 1 4\n");
	printf("-solver 0 -timer -nrecur 1 -o output/\n");
	printf("\n");
      }
      MPI_Finalize();
      return (0);
    }
    
    if (check_pgrid) {
      if (myid ==0) {
	printf("Error: Processor grid does not match the total number of processors. \n");
	printf("       npi * npj * npk must equal num_proc (see -help). \n");
      }
      MPI_Finalize();
      return (0);
    }	
  }

  gsl_rng_set(rstate, seed);

  /* Set dx0, dx1, dx2 */
  model_set_spacing_matrices(&dx0, &dx1, &dx2, ni, nj, nk, npi, npj, npk, x0end);


  // TODO check source_ptr exists and all parameters exist inside
  
  /* Figure out processor grid (npi x npj x npk). Processor position 
     indicated by pi, pj, pk. Size of local grid on processor is 
     (ni x nj x nk) */
  
  pk = myid / (npi * npj);
  pj = (myid - pk * npi * npj) / npi;
  pi = myid - pk * npi * npj - pj * npi;

  /* Figure out the extents of each processor's piece of the grid. */
  ilower[0] = pi * ni;
  ilower[1] = pj * nj;
  ilower[2] = pk * nk;
  
  iupper[0] = ilower[0] + ni - 1;
  iupper[1] = ilower[1] + nj - 1;
  iupper[2] = ilower[2] + nk - 1;

  /* Read parameters from params file  */
  if ( params_ptr == NULL ) {
     printf("No parameter file specified. \n");
     return (0);
  }
  double param_r12;
  double spatial_angle_image[npi * ni][npj * nj];
  double vx[npi * ni][npj * nj];
  double vy[npi * ni][npj * nj];
  double correlation_time_image[npi * ni][npj * nj];
  double correlation_length_image[npi * ni][npj * nj];
  param_read_params_matrices(params_ptr, ni, nj, pi, pj, npi, npj, &param_r12, spatial_angle_image, vx, vy,
                             correlation_time_image, correlation_length_image);

  /* Set up and assemble grid */
  {
    HYPRE_StructGridCreate(MPI_COMM_WORLD, 3, &grid);
    HYPRE_StructGridSetExtents(grid, ilower, iupper);
    
    /* Set periodic boundary conditions of model */
    int bound_con[3];
    model_set_periodic(bound_con, ni, nj, nk, npi, npj, npk, 3);
    HYPRE_StructGridSetPeriodic(grid, bound_con);
    
    HYPRE_StructGridAssemble(grid);
  }
  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Grid initialized: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);

  /* Initialize stencil and Struct Matrix, and set stencil values */
  if (solver_id != 3)
    model_create_stencil(&stencil, 3);
  else
    model_create_stencil_spatial_derivative(&stencil, 3);

  HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A);
  HYPRE_StructMatrixInitialize(A);

  int    nvalues = ni * nj * nk;
  double *values;
  values = (double*) calloc(nvalues, sizeof(double));
  if (solver_id == 3) {
      hdf5_open(source_ptr);
      hdf5_set_directory("/data/");
      hsize_t fdims[3]  = {npk * nk, npj * nj, npi * ni};
      hsize_t fstart[3] = {pk * nk, pj * nj, pi * ni};
      hsize_t fcount[3] = {nk, nj, ni};
      hsize_t mdims[3]  = {nk, nj, ni};
      hsize_t mstart[3] = {0, 0, 0};

      hdf5_read_array(values, "adjoint", 3, fdims, fstart, fcount,
		      mdims, mstart, H5T_NATIVE_DOUBLE);

      model_set_stencil_values_matrices_spatial_angle_derivative(&A, ilower, iupper, ni, nj, npi, npj, nk, pi, pj, pk, dx0, dx1,
			               dx2, param_r12, spatial_angle_image, vx, vy, correlation_time_image, correlation_length_image, values);

      hdf5_read_array(values, "data_raw", 3, fdims, fstart, fcount,
		      mdims, mstart, H5T_NATIVE_DOUBLE);

      hdf5_close();
  } else {
      model_set_stencil_values_matrices(&A, ilower, iupper, ni, nj, npi, npj, nk, pi, pj, pk, dx0, dx1, dx2, param_r12,
			               spatial_angle_image, vx, vy, correlation_time_image, correlation_length_image, solver_id);
  }
  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Stencils values set: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);
	
  /* Fix boundary conditions and assemble Struct Matrix */
  model_set_bound(&A, ni, nj, nk, pi, pj, pk, npi, npj, npk, dx0, dx1, dx2);

  HYPRE_StructMatrixAssemble(A);

  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Boundary conditions set: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);

  /* Set up Struct Vectors for b and x */
  HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
  HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x);

  HYPRE_StructVectorInitialize(b);
  HYPRE_StructVectorInitialize(x);

  if (solver_id != 3) {
    /* Set the source term */
    if ( source_ptr == NULL )
        param_set_source(values, rstate, ni, nj, nk, pi, pj, pk, npi, npj, npk,
                        dx0, dx1, dx2, num_recursions);
    else {
        hdf5_open(source_ptr);

        hdf5_set_directory("/data/");

        hsize_t fdims[3]  = {npk * nk, npj * nj, npi * ni};
        hsize_t fstart[3] = {pk * nk, pj * nj, pi * ni};
        hsize_t fcount[3] = {nk, nj, ni};
        hsize_t mdims[3]  = {nk, nj, ni};
        hsize_t mstart[3] = {0, 0, 0};

        hdf5_read_array(values, "data_raw", 3, fdims, fstart, fcount,
                mdims, mstart, H5T_NATIVE_DOUBLE);

        hdf5_close();
      }
    }
    
  HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);

  for (i = 0; i < nvalues; i ++)
    values[i] = 0.0;
  HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);

  free(values);

  HYPRE_StructVectorAssemble(b);
  HYPRE_StructVectorAssemble(x);

  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Struct vector assembled: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);

  /* Setup output file */
  if ( (myid == 0) && (user_filename == 0) ) {
    param_set_output_name(filename, ni, nj, nk, npi, npj, npk, dir_ptr);
  }
  MPI_Bcast(&filename, 255, MPI_CHAR, 0, MPI_COMM_WORLD);

  // TODO create directory if doesn't exist
  if ( (output) && (constrained == 0) && (vinit == 0)) {
    hdf5_create(filename);
    hdf5_set_directory("/");
    hdf5_make_directory("data");
    hdf5_set_directory("/data/");
  }

  /* Set up and use a struct solver */
  if (solver_id == 0) {
    int num_iterations;

    HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_StructPCGSetMaxIter(solver, maxiter );
    HYPRE_StructPCGSetTol(solver, tol );
    HYPRE_StructPCGSetTwoNorm(solver, 1 );
    HYPRE_StructPCGSetRelChange(solver, 0 );
    HYPRE_StructPCGSetPrintLevel(solver, verbose ); /* print each CG iteration */
    HYPRE_StructPCGSetLogging(solver, 1);
    
    /* Use symmetric SMG as preconditioner */
    HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
    HYPRE_StructSMGSetMemoryUse(precond, 0);
    HYPRE_StructSMGSetMaxIter(precond, 1);
    HYPRE_StructSMGSetTol(precond, 0.0);
    HYPRE_StructSMGSetZeroGuess(precond);
    HYPRE_StructSMGSetNumPreRelax(precond, 1);
    HYPRE_StructSMGSetNumPostRelax(precond, 1);
    
    /* Set the preconditioner and solve */
    HYPRE_StructPCGSetPrecond(solver, HYPRE_StructSMGSolve,
			      HYPRE_StructSMGSetup, precond);

    for (j = 0; j < num_recursions; j++) {
      HYPRE_StructPCGSetup(solver, A, b, x);
      HYPRE_StructPCGSolve(solver, A, b, x);
      
      if (j < num_recursions - 1) {
        int    nvalues = ni * nj * nk;
        double *values;

        values = (double*) calloc(nvalues, sizeof(double));

        HYPRE_StructVectorGetBoxValues(x, ilower, iupper, values);

        if (dump) {
          hsize_t fdims[3]  = {npk * nk, npj * nj, npi * ni};
          hsize_t fstart[3] = {pk * nk, pj * nj, pi * ni};
          hsize_t fcount[3] = {nk, nj, ni};
          hsize_t mdims[3]  = {nk, nj, ni};
          hsize_t mstart[3] = {0, 0, 0};

          char step[255];
          sprintf(step, "step_%d", j);

          hdf5_write_array(values, step, 3, fdims, fstart, fcount,
                   mdims, mstart, H5T_NATIVE_DOUBLE);
        }

        HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);

        for (i = 0; i < nvalues; i ++)
          values[i] = 0.0;
        // TODO see if setting x_new to x_old is better than 0.0
        HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);

        // TODO create option to print successive steps

        free(values);

        HYPRE_StructVectorAssemble(b);
        HYPRE_StructVectorAssemble(x);
      }

        /* Get some info on the run */
      HYPRE_StructPCGGetNumIterations(solver, &num_iterations);
      HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

      if ( (myid == 0) && (verbose > 0) ) {
        printf("\n");
        printf("Iterations = %d\n", num_iterations);
        printf("Final Relative Residual Norm = %g\n", final_res_norm);
        printf("\n");
      }
    }

    /* Clean up */
    HYPRE_StructSMGDestroy(precond);
    HYPRE_StructPCGDestroy(solver);
  }
  
  if (solver_id == 1) {
    int num_iterations;
    
    HYPRE_StructSMGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_StructSMGSetMemoryUse(solver, 0);
    HYPRE_StructSMGSetMaxIter(solver, maxiter);
    HYPRE_StructSMGSetTol(solver, tol);
    HYPRE_StructSMGSetRelChange(solver, 0);
    HYPRE_StructSMGSetNumPreRelax(solver, n_pre);
    HYPRE_StructSMGSetNumPostRelax(solver, n_post);
    /* Logging must be on to get iterations and residual norm info below */
    // TODO fix SMG logging
    HYPRE_StructSMGSetPrintLevel(solver, verbose);
    HYPRE_StructSMGSetLogging(solver, 1);

    /* Setup and solve */
    for (j = 0; j < num_recursions; j++) {
      HYPRE_StructSMGSetup(solver, A, b, x);
      HYPRE_StructSMGSolve(solver, A, b, x);

      if (j < num_recursions - 1) {
        int    nvalues = ni * nj * nk;
        double *values;

        values = (double*) calloc(nvalues, sizeof(double));

        HYPRE_StructVectorGetBoxValues(x, ilower, iupper, values);

        if (dump) {
          hsize_t fdims[3]  = {npk * nk, npj * nj, npi * ni};
          hsize_t fstart[3] = {pk * nk, pj * nj, pi * ni};
          hsize_t fcount[3] = {nk, nj, ni};
          hsize_t mdims[3]  = {nk, nj, ni};
          hsize_t mstart[3] = {0, 0, 0};

          char step[255];
          sprintf(step, "step_%d", j);

          hdf5_write_array(values, step, 3, fdims, fstart, fcount,
                   mdims, mstart, H5T_NATIVE_DOUBLE);
        }

        HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);
	
        for (i = 0; i < nvalues; i ++)
          values[i] = 0.0;
        // TODO see if setting _xnew to x_old is better than 0.0
        HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);

        // TODO create option to print successive steps

        free(values);

        HYPRE_StructVectorAssemble(b);
        HYPRE_StructVectorAssemble(x);
      }
      
      /* Get some info on the run */
      HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
      HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &final_res_norm);


      if ( (myid == 0) && (verbose > 0) ) {
        printf("\n");
        printf("Iterations = %d\n", num_iterations);
        printf("Final Relative Residual Norm = %g\n", final_res_norm);
        printf("\n");
      }	
    }
    
    /* Clean up */
    HYPRE_StructSMGDestroy(solver);
  }

  /* Matrix - vector multiplication */
  if ( (solver_id == 2) || (solver_id == 3) ){
    HYPRE_StructMatrixMatvec(-1.0, A, b, 0.0, x);
  }

  if ( (solver_id == 4) ||  (solver_id == 5)){

    HYPRE_Real* eigenvalues = NULL;
    mv_MultiVectorPtr eigenvectors = NULL;
    mv_MultiVectorPtr initvectors = NULL;
    HYPRE_Real* residuals;
    utilities_FortranMatrix* residualNorms;

    mv_MultiVectorPtr constraints = NULL;
    mv_InterfaceInterpreter* interpreter;
    HYPRE_MatvecFunctions matvec_fn;

    if (myid != 0)
        verbose = 0;

    /* define an interpreter for the Struct interface */
    interpreter = hypre_CTAlloc(mv_InterfaceInterpreter, 1, HYPRE_MEMORY_HOST);
    HYPRE_StructSetupInterpreter( interpreter );
    HYPRE_StructSetupMatvec(&matvec_fn);

    HYPRE_LOBPCGCreate(interpreter, &matvec_fn, (HYPRE_Solver*)&solver);
    HYPRE_LOBPCGSetMaxIter((HYPRE_Solver)solver, maxiter);
    HYPRE_LOBPCGSetTol((HYPRE_Solver)solver, tol);
    HYPRE_LOBPCGSetPrintLevel((HYPRE_Solver)solver, verbose);

    /* Use symmetric SMG as preconditioner */
    if (solver_id == 5) {
        HYPRE_LOBPCGSetPrecondUsageMode((HYPRE_Solver)solver, 1);  /*  use rhs as initial guess for inner pcg iterations */
        HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
        HYPRE_StructSMGSetMaxIter(precond, 1);
        HYPRE_LOBPCGSetPrecond( (HYPRE_Solver)solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                                (HYPRE_Solver)precond);
    }

    int    nvalues = ni * nj * nk;
    double *values;
    values = (double*) calloc(nvalues, sizeof(double));
    hsize_t fdims[3]  = {npk * nk, npj * nj, npi * ni};
    hsize_t fstart[3] = {pk * nk, pj * nj, pi * ni};
    hsize_t fcount[3] = {nk, nj, ni};
    hsize_t mdims[3]  = {nk, nj, ni};
    hsize_t mstart[3] = {0, 0, 0};
    char dataname[255];

    if (constrained == 1) {

      int num_constraints;
      hdf5_open(source_ptr);
      hdf5_set_directory("/params/");
      hdf5_read_single_val(&num_constraints, "num_constraints", H5T_STD_I32LE);
      hdf5_set_directory("/data/");

      mv_TempMultiVector* cns;
      cns = hypre_TAlloc(mv_TempMultiVector, 1, HYPRE_MEMORY_HOST);
      hypre_assert( cns != NULL );
      cns->interpreter = interpreter;
      cns->numVectors = num_constraints;
      cns->vector = hypre_CTAlloc(void*,  num_constraints, HYPRE_MEMORY_HOST);
      hypre_assert( cns->vector != NULL );
      cns->ownsVectors = 1;
      for (j = 0; j < num_constraints; j++) {
        sprintf(dataname, "constraint_%d", j);
        hdf5_read_array(values, dataname, 3, fdims, fstart, fcount,
	          mdims, mstart, H5T_NATIVE_DOUBLE);
        HYPRE_StructVector tmp_vector;
        HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &tmp_vector);
	    HYPRE_StructVectorInitialize(tmp_vector);
        HYPRE_StructVectorSetBoxValues(tmp_vector, ilower, iupper, values);
        HYPRE_StructVectorAssemble(tmp_vector);
        cns->vector[j] = tmp_vector;
      }
      hdf5_close();
      constraints = mv_MultiVectorWrap(interpreter, cns, 1);
    }

    int num_vectors;
	if (vinit == 1) {
      int num_init;
      hdf5_open(source_ptr);
      hdf5_set_directory("/params/");
      hdf5_read_single_val(&num_init, "num_init", H5T_STD_I32LE);
      hdf5_set_directory("/data/");

      num_vectors = (num_recursions > num_init) ? num_recursions : num_init;

      if (num_recursions > num_init) {
        initvectors = mv_MultiVectorCreateFromSampleVector(interpreter, num_recursions, x);
	    mv_MultiVectorSetRandom( initvectors, seed );
      }

      mv_TempMultiVector* cns;
      cns = hypre_TAlloc(mv_TempMultiVector, 1, HYPRE_MEMORY_HOST);
      hypre_assert( cns != NULL );
      cns->interpreter = interpreter;
      cns->numVectors = num_vectors;
      cns->vector = hypre_CTAlloc(void*, num_vectors, HYPRE_MEMORY_HOST);
      hypre_assert( cns->vector != NULL );
      cns->ownsVectors = 1;
      for (j = 0; j < num_vectors; j++) {
        if (j < num_init) {
            sprintf(dataname, "eigenvector_%d", j);
            hdf5_read_array(values, dataname, 3, fdims, fstart, fcount,
                  mdims, mstart, H5T_NATIVE_DOUBLE);
            HYPRE_StructVector tmp_vector;
            HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &tmp_vector);
            HYPRE_StructVectorInitialize(tmp_vector);
            HYPRE_StructVectorSetBoxValues(tmp_vector, ilower, iupper, values);
            HYPRE_StructVectorAssemble(tmp_vector);
            cns->vector[j] = tmp_vector;
        } else {
            mv_TempMultiVector* tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(initvectors);
            cns->vector[j] = tmp -> vector[j-num_init];
        }
      }
      hdf5_close();
      eigenvectors = mv_MultiVectorWrap(interpreter, cns, 1);
	} else {
	    num_vectors = num_recursions;
	    /* eigenvectors - create a multivector */
        eigenvectors = mv_MultiVectorCreateFromSampleVector(interpreter, num_recursions, x);
	    mv_MultiVectorSetRandom( eigenvectors, seed );
	}

    if (((constrained==1) || (vinit==1)) && (output)) {
        hdf5_create(filename);
        hdf5_set_directory("/");
        hdf5_make_directory("data");
        hdf5_set_directory("/data/");
      }

    /* Setup B (diagonal) matrix for std-scaling*/
    HYPRE_StructMatrix  B;
    HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &B);
    HYPRE_StructMatrixInitialize(B);
    model_set_stencil_values_std_scaling(&B, ilower, iupper, ni, nj, npi, npj, nk, pi, pj, pk, param_r12,
			               correlation_time_image, correlation_length_image);
    model_set_bound(&B, ni, nj, nk, pi, pj, pk, npi, npj, npk, dx0, dx1, dx2);
    HYPRE_StructMatrixAssemble(B);
    HYPRE_LOBPCGSetupB((HYPRE_Solver)solver,  (HYPRE_Matrix)B, (HYPRE_Vector)x);


    eigenvalues = hypre_CTAlloc(HYPRE_Real, num_vectors, HYPRE_MEMORY_HOST);
    HYPRE_LOBPCGSetup( (HYPRE_Solver)solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x );
    HYPRE_LOBPCGSolve( (HYPRE_Solver)solver, constraints, eigenvectors, eigenvalues );

    /* eigenvectors - get a pointer */
    mv_TempMultiVector* tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors);
    HYPRE_StructVector* pvx = (HYPRE_StructVector*) (tmp -> vector);

    for (j = 0; j < num_vectors; j++) {
        HYPRE_StructVectorGetBoxValues(pvx[j], ilower, iupper, values);
        sprintf(dataname, "eigenvector_%d", j);
        hdf5_write_array(values, dataname, 3, fdims, fstart, fcount,
                         mdims, mstart, H5T_NATIVE_DOUBLE);
        sprintf(dataname, "eigenvalue_%d", j);
        hdf5_write_single_val(&eigenvalues[j], dataname, H5T_NATIVE_DOUBLE);
        residualNorms = HYPRE_LOBPCGResidualNorms( (HYPRE_Solver)solver );
	    residuals = utilities_FortranMatrixValues( residualNorms );
	    sprintf(dataname, "residual_%d", j);
        hdf5_write_single_val(&residuals[j], dataname, H5T_NATIVE_DOUBLE);
    }

    /* clean-up */
    if (solver_id == 5)
        HYPRE_StructSMGDestroy(precond);
    if (constrained == 1)
        mv_MultiVectorDestroy( constraints );
    if (vinit==1)
        mv_MultiVectorDestroy( initvectors );

    free(values);
    free( eigenvalues );
    HYPRE_LOBPCGDestroy((HYPRE_Solver)solver);
    mv_MultiVectorDestroy( eigenvectors );
    free( interpreter );
  }


  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Solver finished: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);	
  
  /* Output data */
  if (output) {
    /* Get the local raw data */
    int nvalues = ni * nj * nk;
    
    double *raw = (double*)calloc(nvalues, sizeof(double));
    double *env = (double*)calloc(nvalues, sizeof(double));
    
    HYPRE_StructVectorGetBoxValues(x, ilower, iupper, raw);  
  
    /* Find statistics for raw data and envelope */
    double avg_raw = 0.;
    double avg_env = 0.;
    double var_raw = 0.;
    double var_env = 0.;
    double min_raw = raw[0];
    double max_raw = raw[0];
    double min_env;
    double max_env;
    
    double *lc_raw = (double*)calloc(npk * nk, sizeof(double));
    double *lc_env = (double*)calloc(npk * nk, sizeof(double));

    /* Calculate mean and variance for raw data */
    {
      for(i = 0; i < nvalues; i++)
	avg_raw += raw[i];
      
      MPI_Allreduce(MPI_IN_PLACE, &avg_raw, 1, MPI_DOUBLE,
		    MPI_SUM, MPI_COMM_WORLD);
      
      avg_raw /= (npi * npj * npk * nvalues);
      
      /* Second pass for variance */
      for (i = 0; i < nvalues; i++)
	var_raw += (raw[i] - avg_raw) * (raw[i] - avg_raw);
      
      MPI_Allreduce(MPI_IN_PLACE, &var_raw, 1, MPI_DOUBLE,
		    MPI_SUM, MPI_COMM_WORLD);
      
      var_raw /= (npi * npj * npk * nvalues - 1);
    }
    
    /* Add envelope and calculate envelope average */
    {
      int l = 0;
      for (k = 0; k < nk; k++) {
	    for (j = 0; j < nj; j++) {
	        for (i = 0; i < ni; i++) {
	            env[l] = param_env(raw[l], avg_raw, var_raw, i, j, k, ni, nj, nk,
			       pi, pj, pk, dx0, dx1, dx2);

	    avg_env += env[l];
	    
	    l++;
	  }
	}
      }
      
      MPI_Allreduce(MPI_IN_PLACE, &avg_env, 1, MPI_DOUBLE,
		    MPI_SUM, MPI_COMM_WORLD);
      
      avg_env /= (npi * npj * npk * nvalues);    
    }

    min_env = env[0];
    max_env = env[0];

    /* Find min, max, lightcurve, var_env */
    {
      int l = 0;
      double area;
      for (k = pk * nk; k < (pk + 1) * nk; k++) {
	for (j = 0; j < nj; j++) {
	  for (i = 0; i < ni; i++) {
	    area = model_area(i, j, k, ni, nj, nk, pi, pj, pk, dx0, dx1, dx2);

	    lc_raw[k] += raw[l] * area;
	    lc_env[k] += env[l] * area;

	    if (raw[l] < min_raw)
	      min_raw = raw[l];
	    if (raw[l] > max_raw)
	      max_raw = raw[l];
	    
	    if (env[l] < min_env)
	      min_env = env[l];
	    if (env[l] > max_env)
	      max_env = env[l];

	    var_env += (env[l] - avg_env) * (env[l] - avg_env);

	    l++;
	  }
	}
      }
      
      MPI_Allreduce(MPI_IN_PLACE, &var_env, 1, MPI_DOUBLE,
		    MPI_SUM, MPI_COMM_WORLD);      

      var_env /= (npi * npj * npk * nvalues - 1);
    }

    // TODO turn allreduce into reduce for min, max, and var
    // (currently required for hdf5_write_single_val)
    MPI_Allreduce(MPI_IN_PLACE, &min_raw, 1, MPI_DOUBLE,
		  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_raw, 1, MPI_DOUBLE,
		  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &min_env, 1, MPI_DOUBLE,
		  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_env, 1, MPI_DOUBLE,
		  MPI_MAX, MPI_COMM_WORLD);
    if (myid == 0) {
      MPI_Reduce(MPI_IN_PLACE, lc_raw, npk * nk, MPI_DOUBLE,
		 MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, lc_env, npk * nk, MPI_DOUBLE,
		 MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else {
      MPI_Reduce(lc_raw, lc_raw, npk * nk, MPI_DOUBLE,
		 MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(lc_env, lc_env, npk * nk, MPI_DOUBLE,
		 MPI_SUM, 0, MPI_COMM_WORLD);
    }

    /* File i/o */

    /* Save solution to output file*/
    hdf5_set_directory("/");
    hdf5_set_directory("/data/");
    // TODO create further heirarchy in file structure

    /* note: HYPRE has k as the slowest varying, opposite of HDF5 */
    {
      hsize_t fdims[3]  = {npk * nk, npj * nj, npi * ni};
      hsize_t fstart[3] = {pk * nk, pj * nj, pi * ni};
      hsize_t fcount[3] = {nk, nj, ni};
      hsize_t mdims[3]  = {nk, nj, ni};
      hsize_t mstart[3] = {0, 0, 0};
      
      hdf5_write_array(raw, "data_raw", 3, fdims, fstart, fcount,
		       mdims, mstart, H5T_NATIVE_DOUBLE);  
      hdf5_write_array(env, "data_env", 3, fdims, fstart, fcount,
		       mdims, mstart, H5T_NATIVE_DOUBLE);
    }
    
    /* Output lightcurve and parameters */
    {
      hsize_t fdims  = npk * nk;
      hsize_t fstart = 0;
      hsize_t fcount = 0;
      hsize_t mdims  = 0;
      hsize_t mstart = 0;

      if (myid == 0) {
        fcount = npk * nk;
        mdims  = npk * nk;
      }
	
      hdf5_write_array(lc_raw, "lc_raw", 1, &fdims, &fstart, &fcount,
		       &mdims, &mstart, H5T_NATIVE_DOUBLE);
      hdf5_write_array(lc_env, "lc_env", 1, &fdims, &fstart, &fcount,
		       &mdims, &mstart, H5T_NATIVE_DOUBLE);
    }
    
    hdf5_set_directory("/");
    hdf5_make_directory("params");
    hdf5_set_directory("/params/");

    hdf5_write_single_val(&param_x0start, "x0start", H5T_IEEE_F64LE);
    hdf5_write_single_val(&x0end, "x0end", H5T_IEEE_F64LE);
    hdf5_write_single_val(&param_x1start, "x1start", H5T_IEEE_F64LE);
    hdf5_write_single_val(&param_x1end, "x1end", H5T_IEEE_F64LE);
    hdf5_write_single_val(&param_x2start, "x2start", H5T_IEEE_F64LE);
    hdf5_write_single_val(&param_x2end, "x2end", H5T_IEEE_F64LE);
    hdf5_write_single_val(&dx0, "dx0", H5T_IEEE_F64LE);
    hdf5_write_single_val(&dx1, "dx1", H5T_IEEE_F64LE);
    hdf5_write_single_val(&dx2, "dx2", H5T_IEEE_F64LE);

    hdf5_write_single_val(&npi, "npi", H5T_STD_I32LE);
    hdf5_write_single_val(&npj, "npj", H5T_STD_I32LE);
    hdf5_write_single_val(&npk, "npk", H5T_STD_I32LE);
    hdf5_write_single_val(&ni, "ni", H5T_STD_I32LE);
    hdf5_write_single_val(&nj, "nj", H5T_STD_I32LE);
    hdf5_write_single_val(&nk, "nk", H5T_STD_I32LE);
    hdf5_write_single_val(&seed, "seed", H5T_STD_U64LE);

    /* Write additional parameters contained in param_<model_name>.c */
    param_write_params(filename);
    
    hdf5_set_directory("/");
    hdf5_make_directory("stats");
    hdf5_set_directory("/stats/");
    hdf5_write_single_val(&min_raw, "min_raw", H5T_IEEE_F64LE);
    hdf5_write_single_val(&max_raw, "max_raw", H5T_IEEE_F64LE);
    hdf5_write_single_val(&avg_raw, "avg_raw", H5T_IEEE_F64LE);
    hdf5_write_single_val(&var_raw, "var_raw", H5T_IEEE_F64LE);

    hdf5_write_single_val(&min_env, "min_env", H5T_IEEE_F64LE);
    hdf5_write_single_val(&max_env, "max_env", H5T_IEEE_F64LE);
    hdf5_write_single_val(&avg_env, "avg_env", H5T_IEEE_F64LE);
    hdf5_write_single_val(&var_env, "var_env", H5T_IEEE_F64LE);
    
    hdf5_close();

    check_t = clock();
    if ( (myid == 0) && (timer) )
      printf("Data output: t = %lf\n\n",
	     (double)(check_t - start_t) / CLOCKS_PER_SEC);
    
    free(raw);
    free(env);
    free(lc_raw);
    free(lc_env);
  }
  
  /* Free HYPRE memory */
  HYPRE_StructGridDestroy(grid);
  HYPRE_StructStencilDestroy(stencil);
  HYPRE_StructMatrixDestroy(A);
  HYPRE_StructVectorDestroy(b);
  HYPRE_StructVectorDestroy(x);

  /* Free GSL rng state */
  gsl_rng_free(rstate);

  /* Finalize HYPRE */
  HYPRE_Finalize();

  /* Finalize MPI */
  MPI_Finalize();
  
  return (0);
}
