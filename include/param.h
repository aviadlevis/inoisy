#ifndef PARAM
#define PARAM

#include <gsl/gsl_rng.h>

extern const char model_name[];

extern const double param_x0start;
extern const double param_x0end;
extern const double param_x1start;
extern const double param_x1end;
extern const double param_x2start;
extern const double param_x2end;

void param_read_params(char* filename);

void param_read_params_matrices(char* filename, int ni, int nj, int pi, int pj, int npi, int npj,
                              double *param_r12,  double spatial_angle_image[ni][nj],
                              double vx[ni][nj],  double vy[ni][nj],  double correlation_time_image[ni][nj],
                              double correlation_length_image[ni][nj]);

void param_write_params(char* filename);

void param_set_output_name(char* filename, int ni, int nj, int nk,
			   int npi, int npj, int npk, char* dir);

double param_env(double raw, double avg_raw, double var_raw,
		 int i, int j, int k, int ni, int nj, int nk,
		 int pi, int pj, int pk, double dx0, double dx1, double dx2);

void param_coeff(double* coeff, double x0, double x1, double x2, double dx0,
		 double dx1, double dx2, int index);

void param_coeff_matrices(double* coeff, double x0, double x1, double x2, double dx0, double dx1,
                          double dx2, double param_r12, int ni, int nj, double spatial_angle[ni][nj],
                          double vx[ni][nj], double vy[ni][nj], double correlation_time[ni][nj],
                          double correlation_length[ni][nj], int gridi, int gridj);

void param_set_source(double* values, gsl_rng* rstate, int ni, int nj, int nk,
		      int pi, int pj, int pk, int npi, int npj, int npk,
		      double dx0, double dx1, double dx2, int nrecur);


void param_coeff_matrices_spatial_angle_derivative(double* dvalues, double x0, double x1, double x2, double dx0, double dx1, double dx2,
                 double param_r12, int ni, int nj, int nk, double spatial_angle[ni][nj], double vx[ni][nj], double vy[ni][nj],
                 double correlation_time[ni][nj], double correlation_length[ni][nj], int gridi, int gridj, int gridk, double adjoint[nk][nj][ni]);


#endif
