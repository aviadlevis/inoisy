#include "model_general_xy.c"

void model_set_spacing_matrices(double* dx0, double* dx1, double* dx2,
		       int ni, int nj, int nk, int npi, int npj, int npk, double x0end)
{
  *dx0 = (x0end - param_x0start) / (npk * nk);
  *dx1 = (param_x1end - param_x1start) / (npj * nj);
  *dx2 = (param_x2end - param_x2start) / (npi * ni);
}

void model_set_stencil_values_matrices(HYPRE_StructMatrix* A, int* ilower, int* iupper,
			      int ni, int nj, int npi, int npj, int nk, int pi, int pj, int pk, double dx0, double dx1, double dx2,
			      double param_r12, double spatial_angle_image[npi * ni][npj * nj], double vx[npi * ni][npj * nj], double vy[npi * ni][npj * nj],
			      double correlation_time_image[npi * ni][npj * nj], double correlation_length_image[npi * ni][npj * nj])
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
    x1 = param_x1start + dx1 * gridj;
    x2 = param_x2start + dx2 * gridi;

    param_coeff_matrices(coeff, x0, x1, x2, dx0, dx1, dx2, param_r12, ni, nj, npi, npj,
                spatial_angle_image, vx, vy, correlation_time_image, correlation_length_image, gridi, gridj);

    /*0=a, 1=b, 2=c, 3=d, 4=e, 5=f, etc.*/
    /*
      xx 14 xx    10 04 09    xx 18 xx    ^
      11 05 12    01 00 02    15 06 16    |
      xx 13 xx    07 03 08    xx 17 xx    j i ->    k - - >

      22 14 21    10 04 09    26 18 25    ^
      11 05 12    01 00 02    15 06 16    |
      19 13 20    07 03 08    23 17 24    j i ->    k - - >
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

void model_set_stencil_values_matrices_squared(HYPRE_StructMatrix* A, int* ilower, int* iupper,
			      int ni, int nj, int npi, int npj, int nk, int pi, int pj, int pk, double dx0, double dx1, double dx2,
			      double param_r12, double spatial_angle_image[npi * ni][npj * nj], double vx[npi * ni][npj * nj], double vy[npi * ni][npj * nj],
			      double correlation_time_image[npi * ni][npj * nj], double correlation_length_image[npi * ni][npj * nj])
{
  int i, j, entry, idx, idx_t;
  int ii, jj, kk, u, v, w;
  int nentries = 93;
  int nvalues = nentries * ni * nj * nk;
  double *values;
  double *values_sqr;
  int stencil_indices[93];
  int mapping[3][3][3] = {{{-1, 14, -1},
                           {11, 5, 12},
                           {-1, 13, -1}},
                          {{10, 4, 9},
                           {1,  0, 2},
                           {7,  3, 8}},
                          {{-1, 18, -1},
                           {15, 6, 16},
                           {-1, 17, -1}}};

  values = (double*) calloc(NSTENCIL, sizeof(double));
  values_sqr = (double*) calloc(nvalues, sizeof(double));

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
    x1 = param_x1start + dx1 * gridj;
    x2 = param_x2start + dx2 * gridi;

    param_coeff_matrices(coeff, x0, x1, x2, dx0, dx1, dx2, param_r12, ni, nj, npi, npj,
                spatial_angle_image, vx, vy, correlation_time_image, correlation_length_image, gridi, gridj);

    values[0]  = coeff[9];
    values[1]  = coeff[0] - coeff[6];
    values[2]  = coeff[0] + coeff[6];
    values[3]  = coeff[2] - coeff[7];
    values[4]  = coeff[2] + coeff[7];
    values[5]  = coeff[5] - coeff[8];
    values[6]  = coeff[5] + coeff[8];
    values[7]  = coeff[1];
    values[8]  = -coeff[1];
    values[9]  = coeff[1];
    values[10] = -coeff[1];
    values[11] = coeff[3];
    values[12] = -coeff[3];
    values[13] = coeff[4];
    values[14] = -coeff[4];
    values[15] = -coeff[3];
    values[16] = coeff[3];
    values[17] = -coeff[4];
    values[18] = coeff[4];

    /* Compute kernel squared values by correlating the 19 pts stencil with itself to generate a
       5x5x5 dimensional stencil with 93 pts */
    entry = 0;
    for (ii = 0; ii < 5; ii++)
        for (jj = 0; jj < 5; jj++)
            for (kk = 0; kk < 5; kk++) {
                if (((abs(ii) > 1) && (abs(jj) > 1) && (abs(kk) > 0)) ||
                    ((abs(ii) > 0) && (abs(jj) > 1) && (abs(kk) > 1)) ||
                    ((abs(ii) > 1) && (abs(jj) > 0) && (abs(kk) > 1)))
                        continue;
                values_sqr[i+entry] = 0;
                for (u = 0; u < 3; u++)
                    for (v = 0; v < 3; v++)
                        for (w = 0; w < 3; w++) {
                            idx = mapping[w][v][u];
                            if ((ii+u-2 < 3) && (ii+u-2 >= 0) && (jj+v-2< 3) &&
                                (jj+v-2 >= 0) && (kk+w-2< 3) && (kk+w-2 >= 0)) {
                                    idx_t = mapping[kk+w-2][jj+v-2][ii+u-2];
                                    if ((idx >= 0) && (idx_t >= 0))
                                        values_sqr[i+entry] += values[idx] * values[idx_t];
                                }
                        }
                entry++;

            }
  }

  HYPRE_StructMatrixSetBoxValues(*A, ilower, iupper, nentries,
				 stencil_indices, values_sqr);

  free(values);
  free(values_sqr);
}



void model_set_stencil_values_std_scaling(HYPRE_StructMatrix* B, int* ilower, int* iupper,
			      int ni, int nj, int npi, int npj, int nk, int pi, int pj, int pk, double param_r12,
			      double correlation_time[npi * ni][npj * nj], double correlation_length[npi * ni][npj * nj])
{
  int i;
  int nentries = 1;
  int nvalues = ni * nj * nk;
  double *values;
  int stencil_indices[1] = {46};
  double factor, scaling;

  values = (double*) calloc(nvalues, sizeof(double));

  for (i = 0; i < nvalues; i ++) {
    int gridi, gridj, gridk, temp;

    temp = i ;
    gridk = temp / (ni * nj);
    gridj = (temp - ni * nj * gridk) / ni;
    gridi = temp - ni * nj * gridk + (pi - gridj) * ni;
    gridj += pj * nj;
    gridk += pk * nk;

    factor = pow(4. * M_PI, 3. / 2.) * tgamma(2.) / tgamma(2. - 3. / 2. );
    scaling = factor * correlation_time[gridi][gridj] * correlation_length[gridi][gridj]
                * param_r12 * correlation_length[gridi][gridj];
    scaling = fmax( sqrt(scaling), 1.E-10 );
    values[i] = scaling;
  }

  HYPRE_StructMatrixSetBoxValues(*B, ilower, iupper, nentries, stencil_indices, values);

  free(values);
}


void model_set_stencil_values_matrices_spatial_angle_derivative(HYPRE_StructMatrix* A, int* ilower, int* iupper,
			      int ni, int nj, int npi, int npj, int nk, int pi, int pj, int pk, double dx0, double dx1, double dx2, double param_r12,
			      double spatial_angle_image[npi * ni][npj * nj], double vx[npi * ni][npj * nj], double vy[npi * ni][npj * nj],
			      double correlation_time_image[npi * ni][npj * nj], double correlation_length_image[npi * ni][npj * nj],
			      double adjoint[nk][nj][ni])
{
  int i, j, k, i1;
  int nentries = NSTENCIL + 6;
  int nvalues = nentries * ni * nj * nk;
  double *values;
  int stencil_indices[nentries];

  values = (double*) calloc(nvalues, sizeof(double));

  for (j = 0; j < nentries; j++)
    stencil_indices[j] = j;

  double x0, x1, x2;

  double dvalues[nentries];
  int gridi, gridj, gridk;

  /*0=a, 1=b, 2=c, 3=d, 4=e, 5=f, etc.*/
  /*
                                  22
  xx xx xx     xx 14 xx        10 04 09      xx 18 xx    xx xx xx   ^
  xx 23 xx     11 05 12     19 01 00 02 20   15 06 16    xx 24 xx   |
  xx xx xx     xx 13 xx        07 03 08      xx 17 xx    xx xx xx   j i ->    k - - >
                                  21
  */
  for (gridk = 0; gridk < nk; gridk++) {
    for (gridi = 0; gridi < ni; gridi++) {
      for (gridj = 0; gridj < nj; gridj++) {
        x0 = param_x0start + dx0 * gridk;
        x1 = param_x1start + dx1 * gridj;
        x2 = param_x2start + dx2 * gridi;
        param_coeff_matrices_spatial_angle_derivative(dvalues, x0, x1, x2, dx0, dx1, dx2, param_r12, ni, nj, npi, npj, nk,
                    spatial_angle_image, vx, vy, correlation_time_image, correlation_length_image, gridi, gridj, gridk, adjoint);

        k = gridk * ni * nj + gridj * ni + gridi;
        i = nentries * k;

        for (i1 = 0; i1 < nentries; i1++)
            values[i+i1] = dvalues[i1];
      }
    }
  }

  HYPRE_StructMatrixSetBoxValues(*A, ilower, iupper, nentries,
				 stencil_indices, values);

  free(values);
}

void model_create_stencil_spatial_derivative(HYPRE_StructStencil* stencil, int dim)
{
  /*
                                  22
  xx xx xx     xx 14 xx        10 04 09      xx 18 xx    xx xx xx   ^
  xx 23 xx     11 05 12     19 01 00 02 20   15 06 16    xx 24 xx   |
  xx xx xx     xx 13 xx        07 03 08      xx 17 xx    xx xx xx   j i ->    k - - >
                                  21
  Adding 19-24 to compute the spatial angle derivatives
  */
  int entry;
  HYPRE_StructStencilCreate(3, NSTENCIL+6, stencil);
  int offsets[NSTENCIL+6][3] = {{0,0,0},
                      {-1,0,0}, {1,0,0},
                      {0,-1,0}, {0,1,0},
                      {0,0,-1}, {0,0,1},
                      {-1,-1,0}, {1,-1,0},
                      {1,1,0}, {-1,1,0},
                      {-1,0,-1}, {1,0,-1},
                      {0,-1,-1}, {0,1,-1},
                      {-1,0,1}, {1,0,1},
                      {0,-1,1}, {0,1,1},
                      {-2,0,0}, {2,0,0},
                      {0,-2,0}, {0,2,0},
                      {0,0,-2}, {1,1,1}};
  for (entry = 0; entry < NSTENCIL+6; entry++)
    HYPRE_StructStencilSetElement(*stencil, entry, offsets[entry]);
}


void model_create_stencil_squared(HYPRE_StructStencil* stencil, int dim)
{

  int entry, i, j, k;
  int offsets[3];

  HYPRE_StructStencilCreate(dim, 93, stencil);
  entry = 0;
  for (i = -2; i < 3; i++)
    for (j = -2; j < 3; j++)
      for (k = -2; k < 3; k++) {
        if (((abs(i) > 1) && (abs(j) > 1) && (abs(k) > 0)) ||
            ((abs(i) > 0) && (abs(j) > 1) && (abs(k) > 1)) ||
            ((abs(i) > 1) && (abs(j) > 0) && (abs(k) > 1)))
                continue;
        offsets[0] = i;
        offsets[1] = j;
        offsets[2] = k;
        printf("%d: %d, %d, %d\n", entry, i, j, k);
        HYPRE_StructStencilSetElement(*stencil, entry, offsets);
        entry++;
      }

}

