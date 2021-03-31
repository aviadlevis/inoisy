#include "model_general_xy.c"

void model_set_spacing_matrices(double* dx0, double* dx1, double* dx2, int ni, int nj, int nk, int npi,
		        int npj, int npk, double x0end, double x1start, double x1end, double x2start, double x2end)
{
  *dx0 = (x0end - param_x0start) / (npk * nk);
  *dx1 = (x1end - x1start) / (npj * nj);
  *dx2 = (x2end - x2start) / (npi * ni);
}

void model_set_stencil_values_matrices(HYPRE_StructMatrix* A, int* ilower, int* iupper, int ni, int nj, int npi,
			      int npj, int nk, int pi, int pj, int pk, double dx0, double dx1, double dx2,
			      double x1start, double x2start, double param_r12, double spatial_angle_image[npi * ni][npj * nj],
			      double vx[npi * ni][npj * nj], double vy[npi * ni][npj * nj],
			      double correlation_time_image[npi * ni][npj * nj],
			      double correlation_length_image[npi * ni][npj * nj])
{
  int i, j;
  int nentries = NSTENCIL;
  int nvalues = nentries * ni * nj * nk;
  double *values;
  int stencil_indices[NSTENCIL];

  values = (double*) calloc(nvalues, sizeof(double));

  for (j = 0; j < nentries; j++)
    stencil_indices[j] = j;
  
  for (i = 0; i < nvalues; i += nentries) {
    double x0, x1, x2;
    double coeff[10];
    int gridi, gridj, gridk, temp;
    
    temp = i / nentries;
    gridk = temp / (ni * nj);
    gridj = (temp - ni * nj * gridk) / ni;
    gridi = temp - ni * nj * gridk + (pi - gridj) * ni;
    gridj += pj * nj;
    gridk += pk * nk;
    
    x0 = param_x0start + dx0 * gridk;			
    x1 = x1start + dx1 * gridj;
    x2 = x2start + dx2 * gridi;

    param_coeff_matrices(coeff, x0, x1, x2, dx0, dx1, dx2, param_r12, ni, nj, npi, npj,
                spatial_angle_image, vx, vy, correlation_time_image, correlation_length_image, gridi, gridj);

    /*0=a, 1=b, 2=c, 3=d, 4=e, 5=f, etc.*/
    /*
      xx 14 xx    10 04 09    xx 18 xx    ^
      11 05 12    01 00 02    15 06 16    |
      xx 13 xx    07 03 08    xx 17 xx    j i ->    k - - >
    */
    values[i]    = coeff[9];
    values[i+1]  = coeff[0] - coeff[6];
    values[i+2]  = coeff[0] + coeff[6];
    values[i+3]  = coeff[2] - coeff[7];
    values[i+4]  = coeff[2] + coeff[7];
    values[i+5]  = coeff[5] - coeff[8];
    values[i+6]  = coeff[5] + coeff[8];
    values[i+7]  = coeff[1];
    values[i+8]  = -coeff[1];
    values[i+9]  = coeff[1];
    values[i+10] = -coeff[1];
    values[i+11] = coeff[3];
    values[i+12] = -coeff[3];
    values[i+13] = coeff[4];
    values[i+14] = -coeff[4];
    values[i+15] = -coeff[3];
    values[i+16] = coeff[3];
    values[i+17] = -coeff[4];
    values[i+18] = coeff[4];

    /* values[i+19] = 0.; */
    /* values[i+20] = 0.; */
    /* values[i+21] = 0.; */
    /* values[i+22] = 0.; */
    /* values[i+23] = 0.; */
    /* values[i+24] = 0.; */
    /* values[i+25] = 0.; */
    /* values[i+26] = 0.; */
  }
  
  HYPRE_StructMatrixSetBoxValues(*A, ilower, iupper, nentries,
				 stencil_indices, values);

  free(values);
}