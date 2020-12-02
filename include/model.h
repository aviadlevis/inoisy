#ifndef MODEL
#define MODEL

#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_struct_ls.h"
#include "HYPRE_lobpcg.h"

int model_set_gsl_seed(int seed, int myid);

void model_set_spacing(double* dx0, double* dx1, double* dx2,
		       int ni, int nj, int nk, int npi, int npj, int npk);

void model_set_periodic(int* bound, int ni, int nj, int nk,
			int npi, int npj, int npk, int dim);

void model_create_stencil(HYPRE_StructStencil* stencil, int dim);

void model_create_stencil_spatial_derivative(HYPRE_StructStencil* stencil, int dim);

void model_create_stencil_squared(HYPRE_StructStencil* stencil, int dim);

void model_set_stencil_values(HYPRE_StructMatrix* A, int* ilower, int* iupper,
			      int ni, int nj, int nk, int pi, int pj, int pk,
			      double dx0, double dx1, double dx2);

void model_set_stencil_values_matrices(HYPRE_StructMatrix* A, int* ilower, int* iupper,
			      int ni, int nj, int npi, int npj, int nk, int pi, int pj, int pk, double dx0, double dx1, double dx2,
			      double param_r12, double spatial_angle_image[npi * ni][npj * nj], double vx[npi * ni][npj * nj], double vy[npi * ni][npj * nj],
			      double correlation_time_image[npi * ni][npj * nj], double correlation_length_image[npi * ni][npj * nj]);

void model_set_stencil_values_matrices_squared(HYPRE_StructMatrix* A, int* ilower, int* iupper,
			      int ni, int nj, int npi, int npj, int nk, int pi, int pj, int pk, double dx0, double dx1, double dx2,
			      double param_r12, double spatial_angle_image[npi * ni][npj * nj], double vx[npi * ni][npj * nj], double vy[npi * ni][npj * nj],
			      double correlation_time_image[npi * ni][npj * nj], double correlation_length_image[npi * ni][npj * nj]);

void model_set_stencil_values_std_scaling(HYPRE_StructMatrix* B, int* ilower, int* iupper,
			      int ni, int nj, int npi, int npj, int nk, int pi, int pj, int pk, double param_r12,
			      double correlation_time[npi * ni][npj * nj], double correlation_length[npi * ni][npj * nj]);

void model_set_stencil_values_matrices_spatial_angle_derivative(HYPRE_StructMatrix* A, int* ilower, int* iupper,
			      int ni, int nj, int npi, int npj, int nk, int pi, int pj, int pk, double dx0, double dx1, double dx2, double param_r12,
			      double spatial_angle_image[npi * ni][npj * nj], double vx[npi * ni][npj * nj], double vy[npi * ni][npj * nj],
			      double correlation_time_image[npi * ni][npj * nj], double correlation_length_image[npi * ni][npj * nj],
			      double adjoint[nk][nj][ni]);


void model_set_bound(HYPRE_StructMatrix* A, int ni, int nj, int nk,
		     int pi, int pj, int pk, int npi, int npj, int npk,
		     double dx0, double dx1, double dx2);

double model_area(int i, int j, int k, int ni, int nj, int nk,
		  int pi, int pj, int pk, double dx0, double dx1, double dx2);

void model_set_spacing_matrices(double* dx0, double* dx1, double* dx2,
		       int ni, int nj, int nk, int npi, int npj, int npk, double x0end);
#endif
