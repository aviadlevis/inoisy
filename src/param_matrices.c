#include "param_general_xy.c"


/* Read in parameters from input file */
void param_read_params_matrices(char* filename, int ni, int nj, int pi, int pj, int npi, int npj,
                               double *param_r12,  double spatial_angle_image[npi * ni][npj * nj],
                               double vx[npi * ni][npj * nj],  double vy[npi * ni][npj * nj], double correlation_time_image[npi * ni][npj * nj],
                               double correlation_length_image[npi * ni][npj * nj])
{
    /* hdf5_utils accesses a single global file at a time */
    hdf5_open(filename);

    /* Read diffusion parameters */
    hdf5_set_directory("/diffusion/");
    hsize_t fdims[2]  = {npj * nj, npi * ni};
    hsize_t fstart[2] = {0, 0};
    hsize_t fcount[2] = {npj * nj, npi * ni};
    hsize_t mdims[2]  = {npj * nj, npi * ni};
    hsize_t mstart[2] = {0, 0};
    hdf5_read_array(spatial_angle_image, "spatial_angle", 2, fdims, fstart, fcount,
        mdims, mstart, H5T_NATIVE_DOUBLE);
    hdf5_read_array(correlation_time_image, "correlation_time", 2, fdims, fstart, fcount,
        mdims, mstart, H5T_NATIVE_DOUBLE);
    hdf5_read_array(correlation_length_image, "correlation_length", 2, fdims, fstart, fcount,
        mdims, mstart, H5T_NATIVE_DOUBLE);
    hdf5_read_single_val(param_r12, "tensor_ratio", H5T_IEEE_F64LE);

    /* Read 'advection' velocity parameters */
    hdf5_set_directory("/advection/");
    hdf5_read_array(vx, "vx", 2, fdims, fstart, fcount,
        mdims, mstart, H5T_NATIVE_DOUBLE);
    hdf5_read_array(vy, "vy", 2, fdims, fstart, fcount,
        mdims, mstart, H5T_NATIVE_DOUBLE);

    hdf5_close();
}

int min(int x, int y)
{
  return (x < y) ? x : y;
}

int max(int x, int y)
{
  return (x > y) ? x : y;
}

/* time correlation vector (1, v1, v2) */
static void set_u0_matrices(double* u0, double x0, double x1, double x2, double vx, double vy)
{
  u0[0] = 1.;
  u0[1] = vx;
  u0[2] = vy;
}

/* unit vectors in direction of major and minor axes */
static void set_u1_u2_matrices(double* u1, double* u2, double x0, double x1, double x2, double theta)
{
  u1[0] = 0.;
  u2[0] = 0.;

  u1[1] = cos(theta);
  u1[2] = sin(theta);

  u2[1] = -sin(theta);
  u2[2] = cos(theta);

}

static void set_h_matrices(double h[3][3], double x0, double x1, double x2, double param_r12,
                  double spatial_angle, double vx, double vy, double correlation_time, double correlation_length)
{
  int i, j;
  double u0[3], u1[3], u2[3];

  set_u0_matrices(u0, x0, x1, x2, vx, vy);
  set_u1_u2_matrices(u1, u2, x0, x1, x2, spatial_angle);
  
  double lam0, lam1, lam2; /* temporal, major, minor correlation lengths */

  lam0 = correlation_time;
  lam1 = correlation_length;
  lam2 = param_r12 * lam1;

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      h[i][j] = lam0 * lam0 * u0[i] * u0[j]
	            + lam1 * lam1 * u1[i] * u1[j]
	            + lam2 * lam2 * u2[i] * u2[j];
    }
  }
}


/* dh[0][2][1] = dh[0][2]/dx[1] */
static void set_dh_matrices(double dh[][3][3], double x0, double x1, double x2,
		   double dx0, double dx1, double dx2, double param_r12, int ni, int nj, int npi, int npj, double spatial_angle[npi * ni][npj * nj],
		   double vx[npi * ni][npj * nj], double vy[npi * ni][npj * nj], double correlation_time[npi * ni][npj * nj],
		   double correlation_length[npi * ni][npj * nj], int gridi, int gridj)
{
  int i, j, k;

  double dx[3] = {dx0, dx1, dx2};
  
  /* hm[0][2][1] = h(x0 - dx0, x1, x2)[2][1] */
  double hm[3][3][3], hp[3][3][3];
  set_h_matrices(hm[0], x0 - dx0, x1, x2, param_r12, spatial_angle[min(max(gridi,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)], vx[min(max(gridi,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)], vy[min(max(gridi,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)], correlation_time[min(max(gridi,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)], correlation_length[min(max(gridi,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)]);
  set_h_matrices(hp[0], x0 + dx0, x1, x2, param_r12, spatial_angle[min(max(gridi,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)], vx[min(max(gridi,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)], vy[min(max(gridi,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)], correlation_time[min(max(gridi,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)], correlation_length[min(max(gridi,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)]);
  set_h_matrices(hm[1], x0, x1 - dx1, x2, param_r12, spatial_angle[min(max(gridi,0), npi*ni-1)][min(max(gridj-1,0), npj*nj-1)], vx[min(max(gridi,0), npi*ni-1)][min(max(gridj-1,0), npj*nj-1)], vy[min(max(gridi,0), npi*ni-1)][min(max(gridj-1,0), npj*nj-1)], correlation_time[min(max(gridi,0), npi*ni-1)][min(max(gridj-1,0), npj*nj-1)], correlation_length[min(max(gridi,0), npi*ni-1)][min(max(gridj-1,0), npj*nj-1)]);
  set_h_matrices(hp[1], x0, x1 + dx1, x2, param_r12, spatial_angle[min(max(gridi,0), npi*ni-1)][min(max(gridj+1,0), npj*nj-1)], vx[min(max(gridi,0), npi*ni-1)][min(max(gridj+1,0), npj*nj-1)], vy[min(max(gridi,0), npi*ni-1)][min(max(gridj+1,0), npj*nj-1)], correlation_time[min(max(gridi,0), npi*ni-1)][min(max(gridj+1,0), npj*nj-1)], correlation_length[min(max(gridi,0), npi*ni-1)][min(max(gridj+1,0), npj*nj-1)]);
  set_h_matrices(hm[2], x0, x1, x2 - dx2, param_r12, spatial_angle[min(max(gridi-1,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)], vx[min(max(gridi-1,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)], vy[min(max(gridi-1,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)], correlation_time[min(max(gridi-1,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)], correlation_length[min(max(gridi-1,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)]);
  set_h_matrices(hp[2], x0, x1, x2 + dx2, param_r12, spatial_angle[min(max(gridi+1,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)], vx[min(max(gridi+1,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)], vy[min(max(gridi+1,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)], correlation_time[min(max(gridi+1,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)], correlation_length[min(max(gridi+1,0), npi*ni-1)][min(max(gridj,0), npj*nj-1)]);

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      for (k = 0; k < 3; k++)
	    dh[i][j][k] = 0.5 * ( hp[k][i][j] - hm[k][i][j] ) / dx[k];
}


void param_coeff_matrices(double* coeff, double x0, double x1, double x2, double dx0, double dx1, double dx2,
                 double param_r12, int ni, int nj, int npi, int npj, double spatial_angle[npi * ni][npj * nj], double vx[npi * ni][npj * nj], double vy[npi * ni][npj * nj],
                 double correlation_time[npi * ni][npj * nj], double correlation_length[npi * ni][npj * nj], int gridi, int gridj)
{
  double h[3][3], dh[3][3][3];
  set_h_matrices(h, x0, x1, x2, param_r12, spatial_angle[gridi][gridj], vx[gridi][gridj], vy[gridi][gridj], correlation_time[gridi][gridj], correlation_length[gridi][gridj]);
  set_dh_matrices(dh, x0, x1, x2, dx0, dx1, dx2, param_r12, ni, nj, npi, npj, spatial_angle, vx, vy, correlation_time, correlation_length, gridi, gridj);

  /* dy^2 */
  coeff[0] = -h[2][2] / (dx2 * dx2);
  /* dydx */
  coeff[1] = -0.5 * h[1][2] / (dx1 * dx2);
  /* dx^2 */
  coeff[2] = -h[1][1] / (dx1 * dx1);
  /* dydt */
  coeff[3] = -0.5 * h[0][2] / (dx0 * dx2);
  /* dxdt */
  coeff[4] = -0.5 * h[0][1] / (dx0 * dx1);
  /* dt^2 */
  coeff[5] = -h[0][0] / (dx0 * dx0);
  /* dy */
  coeff[6] = -0.5 * ( dh[0][2][0] + dh[1][2][1] + dh[2][2][2] ) / dx2;
  /* dx */
  coeff[7] = -0.5 * ( dh[0][1][0] + dh[1][1][1] + dh[2][1][2] ) / dx1;
  /* dt */
  coeff[8] = -0.5 * ( dh[0][0][0] + dh[1][0][1] + dh[2][0][2] ) / dx0;
  /* const */
  coeff[9] = -2. * ( coeff[0] + coeff[2] + coeff[5] ) + ksq(x0, x1, x2);
}



