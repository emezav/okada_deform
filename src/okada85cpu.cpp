/**
 * @file
 * @brief CPU implementation of Okada deformation model
 * Okada Y., Surface deformation due to shear and tensile faults in a  half-space
 * Bull. Seismol. Soc. Am., 75:4, 1135-1154, 1985.
 * @author Erwin Meza Vega <emezav@unicauca.edu.co>, <emezav@gmail.com>
 * @copyright MIT License
 */

#include "grid.h"
#include "okada85cpu.h"

namespace okada85cpu
{
  // Begin namespace

  using namespace grid;
  using namespace okada85;

  /**
   * @brief Checks if the absolute value ot the parameter is less than epsilon
   * @param val Value to check
   * @param eps Epsilon (Small value close to zero)
   * @return True if the absolute value is less than epsilon, false otherwise.
   */
  inline bool isZero(float val, float eps)
  {
    return (fabsf(val) < eps);
  }

  /**
   * @brief Calculates Chinnery's strike / slip (25) PP. 1144 for (24) PP. 1143
   * @param U1 strike-slip component of the dislocation (Fig. 1) PP. 1138
   * @param Mu_L Mu-Lambda coefficient for I1 - I5
   * @param x Outer integral variable (23) PP. 1143
   * @param p Inner integral variable (23) PP. 1143
   * @param q y*sin(dip) - d*cos(dip) (30) PP. 1145
   * @param cs cos(dip) to avoid recalculation
   * @param sn sin(dip) to avoid recalculation
   * @param L Fault length (Fig. 1) PP. 1138
   * @param W Fault width (Fig. 1) PP. 1138
   * @param Eps Parameter to check if a value is close to zero.
   * @param Ux (Output) Reference to store the result for the x component (26)
   * @param Uy (Output) Reference to store the result for the y component (26)
   * @param Uz (Output) Reference to store the result for the z component (26)
   */
  void chinneryStrikeSlip(
      float U1,
      float Mu_L,
      float x,
      float p,
      float q,
      float cs,
      float sn,
      float L,
      float W,
      float Eps,
      float &Ux,
      float &Uy,
      float &Uz);

  /**
   * @brief Calculates Chinnery's (24) PP. 1143 for dip (26) PP. 1144
   * @param U2 dip-slip component of the dislocation (Fig. 1) PP. 1138
   * @param Mu_L Mu-Lambda coefficient for I1 - I5
   * @param x Outer integral variable (23) PP. 1143
   * @param p Inner integral variable (23) PP. 1143
   * @param q y*sin(dip) - d*cos(dip) (30) PP. 1145
   * @param cs cos(dip) to avoid recalculation
   * @param sn sin(dip) to avoid recalculation
   * @param L Fault length (Fig. 1) PP. 1138
   * @param W Fault width (Fig. 1) PP. 1138
   * @param Ux (Output) Reference to store the result for the x component (26)
   * @param Uy (Output) Reference to store the result for the y component (27)
   * @param Uz (Output) Reference to store the result for the z component (28)
   * @param Eps Parameter to check if a value is close to zero.
   */
  void chinneryDipSlip(
      float U2,
      float Mu_L,
      float x,
      float p,
      float q,
      float cs,
      float sn,
      float L,
      float W,
      float Eps,
      float &Ux,
      float &Uy,
      float &Uz);

  /**
   * @brief Calculates the inner part [...] for ux, uy, uz for strike-slip given Xi, Eta (25) PP. 1144
   *
   */
  void strikeSlip(
      float Xi,
      float Eta,
      float Mu_L,
      float p,
      float q,
      float cs,
      float sn,
      float L,
      float W,
      float Eps,
      float &Fx,
      float &Fy,
      float &Fz);

  /**
   * @brief Calculates the inner part [...] for ux, uy, uz for dip-slip given Xi, Eta (25) PP. 1144
   *
   */
  void dipSlip(
    float Xi,
      float Eta,
      float Mu_L,
      float p,
      float q,
      float cs,
      float sn,
      float L,
      float W,
      float Eps,
      float &Fx,
      float &Fy,
      float &Fz);

  /**
   * @brief I1 function (28), (29), PP. 1144, 1145
   * @param Xi Xi from Chinery's notation (24) PP. 1143
   * @param Eta Eta from Chinery's notation (24) PP.1143
   * @param Mu_L Mu - Lambda coefficient to calculate I1 (28, 29) PP. 1144, 1145
   * @param R sqrt(Xi^2 + Eta^2 + q^2) (30) PP. 1145
   * @param dTilde eta*sin(dip) - q*cos(dip) (30) PP.1143
   * @param sn sin(dip) to avoid recalculation
   * @param cs cos(dip) to avoid recalculation
   * @param q y*sin(dip) - d*cos(dip) (30) PP. 1145
   * @param i5 Value of I5(Xi, Eta, ...)
   * @param Eps Parameter to check if a value is close to zero.
   */
  float I1(
      float Xi,
      float Eta,
      float Mu_L,
      float R,
      float dTilde,
      float sn,
      float cs,
      float q,
      float i5,
      float Eps);

  /**
   * @brief I2 function (28), (29) PP. 1144, 1145
   * @param Xi Xi from Chinery's notation (24) PP. 1143
   * @param Eta Eta from Chinery's notation (24) PP.1143
   * @param Mu_L Mu - Lambda coefficient to calculate I2 (28, 29) PP. 1144, 1145
   * @param R sqrt(Xi^2 + Eta^2 + q^2) (30) PP. 1145
   * @param yTilde eta*cos(dip) - q*sin(dip) (30) PP.1143
   * @param dTilde eta*sin(dip) - q*cos(dip) (30) PP.1143
   * @param sn sin(dip) to avoid recalculation
   * @param cs cos(dip) to avoid recalculation
   * @param q y*sin(dip) - d*cos(dip) (30) PP. 1145
   * @param i3 value of I3(Xi, Eta, ...)
   * @param Eps Parameter to check if a value is close to zero.
   */
  float I2(
      float Xi,
      float Eta,
      float Mu_L,
      float R,
      float yTilde,
      float dTilde,
      float sn,
      float cs,
      float q,
      float i3,
      float Eps);

  /**
   * @brief I3 function (28), (29) PP. 1144, 1145
   * @param Xi Xi from Chinery's notation (24) PP. 1143
   * @param Eta Eta from Chinery's notation (24) PP.1143
   * @param Mu_L Mu - Lambda coefficient to calculate I3 (28, 29) PP. 1144, 1145
   * @param R sqrt(Xi^2 + Eta^2 + q^2) (30) PP. 1145
   * @param yTilde eta*cos(dip) - q*sin(dip) (30) PP.1143
   * @param dTilde eta*sin(dip) - q*cos(dip) (30) PP.1143
   * @param sn sin(dip) to avoid recalculation
   * @param cs cos(dip) to avoid recalculation
   * @param q y*sin(dip) - d*cos(dip) (30) PP. 1145
   * @param i4 Value of I4(Xi, Eta, ...)
   * @param Eps Parameter to check if a value is close to zero.
   */
  float I3(
      float Xi,
      float Eta,
      float Mu_L,
      float R,
      float yTilde,
      float dTilde,
      float sn,
      float cs,
      float q,
      float i4,
      float Eps);

  /**
   * @brief I4 function (28), (29) PP. 1144, 1145
   * @param Xi Xi from Chinery's notation (24) PP. 1143
   * @param Eta Eta from Chinery's notation (24) PP.1143
   * @param Mu_L Mu - Lambda coefficient to calculate I4 (28, 29) PP. 1144, 1145
   * @param R sqrt(Xi^2 + Eta^2 + q^2) (30) PP. 1145
   * @param dTilde eta*sin(dip) - q*cos(dip) (30) PP.1143
   * @param sn sin(dip) to avoid recalculation
   * @param cs cos(dip) to avoid recalculation
   * @param q y*sin(dip) - d*cos(dip) (30) PP. 1145
   * @param Eps Parameter to check if a value is close to zero.
   */

  float I4(
      float Xi,
      float Eta,
      float Mu_L,
      float R,
      float dTilde,
      float sn,
      float cs,
      float q,
      float Eps);
  /**
   * @brief I5 function (28), (29) PP. 1144, 1145
   * @param Xi Xi from Chinery's notation (24) PP. 1143
   * @param Eta Eta from Chinery's notation (24) PP.1143
   * @param Mu_L Mu - Lambda coefficient to calculate I5 (28, 29) PP. 1144, 1145
   * @param R sqrt(Xi^2 + Eta^2 + q^2) (30) PP. 1145
   * @param dTilde eta*sin(dip) - q*cos(dip) (30) PP.1143
   * @param sn sin(dip) to avoid recalculation
   * @param cs cos(dip) to avoid recalculation
   * @param q y*sin(dip) - d*cos(dip) (30) PP. 1145
   * @param Eps Parameter to check if a value is close to zero.
   */
  float I5(
      float Xi,
      float Eta,
      float Mu_L,
      float R,
      float dTilde,
      float sn,
      float cs,
      float q,
      float Eps);

  status calculateDeform(
      int rows,
      int columns,
      float x0lon,
      float y0lat,
      float dx,
      float dy,
      fault *components,
      int nComponents,
      parameters params,
      float *Uz,
      float *Us,
      float *Ud,
      float *Ux,
      float *Uy)
  {

    // Get deformation parameters
    auto [vp, vs, re, e2, Eps] = params;

    // Proxy variables
    float vs2 = params.vs * params.vs; // S-wave speed
    float vp2 = params.vp * params.vp; // P-wave speed

    // Mu_L Coefficient for I1 - I5 (28), (29)
    float Mu_L = vs2 / (vp2 - vs2);


    // Calculate the distance of 1 arc secont at the latitude
    // of the origin of the grid
    auto [xdst, ydst] = arcSecMeters(y0lat);
    
    // drx - dx in arc seconds
    float drx = dx / xdst;

    // dry - dy in arc seconds
    float dry = dy / ydst;

    // For each one of the components
    for (int n = 0; n < nComponents; n++)
    {
      // Get parameters from the fault
      auto [lon, lat, d, length, width, st, di, sl, hh] = components[n];

      /*
      cout
           << "Fault parameters: "
          << lon
          << " " << lat
          << " " << d
          << " " << length
          << " " << width
          << " " << st
          << " " << di
          << " " << sl
          << " " << hh << endl;
        */

      // Convert strike, dip and slip angles to radians
      float str = radians(90.0f - st); // st (north up) to rad
      float dir = radians(di);         // dip to rad
      float slr = radians(sl);         // slip to rad

      // Calculate sin and cosine of dip angle
      float cs = cosf(dir);
      float sn = sinf(dir);

      float csStr = cosf(str);
      float snStr = sinf(str);

      // Calculate distance from height and angle
      float de = hh + width * sn;

      // Calculate U1, U2, and U3 components
      float U1 = d * cosf(slr);
      float U2 = d * sinf(slr);
      float U3 = d; // Dislocation

      // Calculate i, j fault position on the grid relative to the grid origin
      int i0 = (lon - x0lon) * 3600.0 / drx;
      int j0 = (lat - y0lat) * 3600.0 / dry;

      // j : from the bottom to the top of the grid
      for (int j = rows - 1; j >= 0; j--)
      {
        // i: from the left to the right of the grid
        for (int i = 0; i < columns; i++)
        {
          // Transform into Okada coordinate system.

          /*
           * Text fragment from PP. 1138
           * Elastic medium occupies the region of z =<0 and x axis is taken to be parallel to the strike
           * direction of the fault. Further, we define elementary dislocations U1, U2, and U3 so
           * as to correspond to strike-slip, dip-slip, and tensile components of arbitrary dislocation.
           */

           // X0, y0: Position of the point grid (i, j) relative to i0, j0 in meters
          float x0 = (i - i0) * dx;
          float y0 = (j - j0) * dy;

          // Transform into Okada's coordinate system
          float x = x0 / csStr + (y0 - x0 * tanf(str)) * sinf(str);
          float y = y0 * csStr - x0 * snStr + width * cs;

          // Definitions for p and q from (30) PP. 1145
          float p = y * cs + de * sn;
          float q = y * sn - de * cs;

          // Calculate the strike components for ux, uy and uz (25) PP. 1144 using Chinnery's notation f(Xi, Eta)
          float uxStr, uyStr, uzStr;
          chinneryStrikeSlip(U1, Mu_L, x, p, q, cs, sn, length, width, Eps, uxStr, uyStr, uzStr);

          // Calculate the dip components for ux, uy and uz (26) PP. 1144 using Chinnery's notation f(Xi, Eta)
          float uxDip, uyDip, uzDip;
          chinneryDipSlip(U2, Mu_L, x, p, q, cs, sn, length, width, Eps, uxDip, uyDip, uzDip);

          // NOTE: Tensile components (27) PP.1144 aren't calculated.
          // Implement yourself, or contact me in case you need help!

          // Calculate linear position on the array
          // (row * columns) + column
          // int pos = (j * columns) + i;
          int pos = linear2D(j, i, columns);

          // Add to z
          Uz[pos] += uzStr + uzDip;

          // Add to zs
          Us[pos] += uxStr + uxDip;

          // Add to zd
          Ud[pos] += uyStr + uyDip;

          // Add to zx
          Ux[pos] = Us[pos] * csStr - Ud[pos] * snStr;

          // Add to zy
          Uy[pos] = Us[pos] * snStr + Ud[pos] * csStr;
        }
      } // For each element on the grid
    } // For each one of the components

    return status::SUCCESS;
  }

  void chinneryStrikeSlip(
      float U1,
      float Mu_L,
      float x,
      float p,
      float q,
      float cs,
      float sn,
      float L,
      float W,
      float Eps,
      float &Ux,
      float &Uy,
      float &Uz)
  {

    // Calculates Chinnery's (24) PP. 1143 for strike-slip (25) PP. 1144
    // f(Xi, Eta) = f(x, p) - f(x, p - W) - f(x-L, p) + f(x - L, p - W)
    // Strike must be evaluated for 4 values of Xi, Eta
    // and calculates x, y, z component for each f.
    // Xi = x,     Eta = p      (f1): calculates fx1, fy1, fz1
    // Xi = x,     Eta = p - W  (f2): calculates fx2, fy2, fz2
    // Xi = x - L, Eta = p      (f3): calculates fx3, fy3, fz3
    // Xi = x - L, Eta = p - W  (f4): calculates fx4, fy4, fz4

    // X component of strike-slip to replace on (24)
    float fx1{0};
    float fx2{0};
    float fx3{0};
    float fx4{0};

    // Y component of strike-slip to replace on (24)
    float fy1{0};
    float fy2{0};
    float fy3{0};
    float fy4{0};

    // Z component of strike-slip to replace on (24)
    float fz1{0};
    float fz2{0};
    float fz3{0};
    float fz4{0};

    // f1: strikeSlip(x, p) (25) PP. 1144
    strikeSlip(x, p, Mu_L, p, q, cs, sn, L, W, Eps, fx1, fy1, fz1);
    // f2: strikeSlip(x, p - W) (25) PP. 1144
    strikeSlip(x, p - W, Mu_L, p, q, cs, sn, L, W, Eps, fx2, fy2, fz2);
    // f3: strikeSlip(x -L, p) (25) PP. 1144
    strikeSlip(x - L, p, Mu_L, p, q, cs, sn, L, W, Eps, fx3, fy3, fz3);
    // f4: strikeSlip(x -L, p - W) (25) PP. 1144
    strikeSlip(x - L, p - W, Mu_L, p, q, cs, sn, L, W, Eps, fx4, fy4, fz4);

    // Evaluate chinnery notation for components x, y and z
    Ux = -(U1 / (2.0 * pi)) * (fx1 - fx2 - fx3 + fx4);
    Uy = -(U1 / (2.0 * pi)) * (fy1 - fy2 - fy3 + fy4);
    Uz = -(U1 / (2.0 * pi)) * (fz1 - fz2 - fz3 + fz4);
  }

  void chinneryDipSlip(
      float U2,
      float Mu_L,
      float x,
      float p,
      float q,
      float cs,
      float sn,
      float L,
      float W,
      float Eps,
      float &Ux,
      float &Uy,
      float &Uz)
  {
    // Calculates Chinnery's (24) PP. 1143 for dip-slip  (26) PP. 1144
    // f(Xi, Eta) = f(x, p) - f(x, p - W) - f(x-L, p) + f(x - L, p - W)
    // Strike must be evaluated for 4 values of Xi, Eta
    // and calculates x, y, z component for each f.
    // Xi = x,     Eta = p      (f1): calculates fx1, fy1, fz1
    // Xi = x,     Eta = p - W  (f2): calculates fx2, fy2, fz2
    // Xi = x - L, Eta = p      (f3): calculates fx3, fy3, fz3
    // Xi = x - L, Eta = p - W  (f4): calculates fx4, fy4, fz4

    // X component of dip-slip to replace on (24)
    float fx1{0};
    float fx2{0};
    float fx3{0};
    float fx4{0};

    // Y component of dip-slip to replace on (24)
    float fy1{0};
    float fy2{0};
    float fy3{0};
    float fy4{0};

    // Z component of dip-slip to replace on (24)
    float fz1{0};
    float fz2{0};
    float fz3{0};
    float fz4{0};

    // f1: dipSlip(x, p) (26) PP. 1144
    dipSlip(x, p, Mu_L, p, q, cs, sn, L, W, Eps, fx1, fy1, fz1);
    // f2: dipSlip(x, p - W) (26) PP. 1144
    dipSlip(x, p - W, Mu_L, p, q, cs, sn, L, W, Eps, fx2, fy2, fz2);
    // f3: dipSlip(x -L, p) (26) PP. 1144
    dipSlip(x - L, p, Mu_L, p, q, cs, sn, L, W, Eps, fx3, fy3, fz3);
    // f4: dipSlip(x -L, p - W) (26) PP. 1144
    dipSlip(x - L, p - W, Mu_L, p, q, cs, sn, L, W, Eps, fx4, fy4, fz4);

    // Evaluate chinnery notation for components x, y and z
    Ux = -(U2 / (2.0 * pi)) * (fx1 - fx2 - fx3 + fx4);
    Uy = -(U2 / (2.0 * pi)) * (fy1 - fy2 - fy3 + fy4);
    Uz = -(U2 / (2.0 * pi)) * (fz1 - fz2 - fz3 + fz4);
  }

  void strikeSlip(
      float Xi,
      float Eta,
      float Mu_L,
      float p,
      float q,
      float cs,
      float sn,
      float L,
      float W,
      float Eps,
      float &Fx,
      float &Fy,
      float &Fz)

  {

    // Calculates the inner part [...] for ux, uy, uz for strike-slip on given Xi, Eta (25) PP. 1144

    // Singularities when: (PP. 1148)
    // q -> 0: set atanf((Xi * Eta)/(q * R)) to 0 in equation (25)
    // Xi -> 0: set I5 = 0 (handled by I5)
    // (R + Eta) -> 0 all terms which contain (R + Eta) on denominator are set to 0
    // and replace ln(R + Eta) to -ln(R + Eta) in equations (28)

    // Calculate yTilde (30) PP. 1145
    float yTilde = Eta * cs + q * sn;

    // Calculate dTilde (30) PP. 1145
    float dTilde = Eta * sn - q * cs;

    // Calculate R (30) PP. 1145
    float R = sqrtf(Xi * Xi + Eta * Eta + q * q);

    // Calculate I1 - I5, (28) (29) PP. 1145
    // Order of calculation: I5, I4, I3 (uses I4), I2 (uses I3), I1 (Uses I5)
    // Each function checks for singularities and return 0 if calculation is not possible.

    float i5 = I5(Xi, Eta, Mu_L, R, dTilde, sn, cs, q, Eps);
    float i4 = I4(Xi, Eta, Mu_L, R, dTilde, sn, cs, q, Eps);
    float i3 = I3(Xi, Eta, Mu_L, R, yTilde, dTilde, sn, cs, q, i4, Eps);
    float i2 = I2(Xi, Eta, Mu_L, R, yTilde, dTilde, sn, cs, q, i3, Eps);
    float i1 = I1(Xi, Eta, Mu_L, R, dTilde, sn, cs, q, i5, Eps);

    // Fx, Fy, Fz: inner part of equation [ ... ]|| (25) PP. 1144 for ux, uy and uz

    // Initialize Fx, Fy, Fz to the last term of (25) PP. 1144 to return early
    // in case of zero or singularity
    // Fx : I1 * sn
    // Fy : I2 * sn
    // Fz : I4 * sn
    // I1, I2 and I4 return zero if any singularity exists.
    Fx = i1 * sn;
    Fy = i2 * sn;
    Fz = i4 * sn;

    // Terms are evaluated to zero or singularity exists:
    // If q is zero, first and second terms of Fx, Fy and Fz are set to zero:
    // On Fx, q sets the numerator to zero on the first term and singularity on the second term.
    // On Fy and Fz, q sets the numerator to zero on both first and second term.
    if (isZero(q, Eps))
    {
      return;
    }

    // Flags to check singularities on R and (R + Eta)
    bool zeroR = isZero(R, Eps);
    bool zeroREta = isZero(R + Eta, Eps);

    // If Both R and (R + Eta) are not zero, add first terms to Fx, Fy and Fz
    if (!zeroR && !zeroREta)
    {
      // First term of Fx
      Fx += (Xi * q) / (R * (R + Eta));
      // First term of Fy
      Fy += (yTilde * q) / (R * (R + Eta));
      // First term of Fz
      Fz += (dTilde * q) / (R * (R + Eta));
    }

    // If R is not zero (q already checked), add the second term to Fx
    if (!zeroR)
    {
      Fx += atanf((Xi * Eta) / (q * R));
    }

    // If  (R + Eta) is not zero, add second term to Fy and Fz
    if (!zeroREta)
    {
      Fy += (q * cs) / (R + Eta);
      Fz += (q * sn) / (R + Eta);
    }
  }

  void dipSlip(
      float Xi,
      float Eta,
      float Mu_L,
      float p,
      float q,
      float cs,
      float sn,
      float L,
      float W,
      float Eps,
      float &Fx,
      float &Fy,
      float &Fz)
  {
    // Calculates the inner part [...] for ux, uy, uz for dip-slip on given Xi, Eta (25) PP. 1144

    // Singularities when: (PP. 1148)
    // q -> 0: set atanf((Xi * Eta)/(q * R)) to 0 in equation (26)
    // Xi -> 0: set I5 = 0 (handled by I5)
    // (R + Eta) -> 0 all terms which contain (R + Eta) on denominator are set to 0
    // and replace ln(R + Eta) to -ln(R + Eta) in equations (28)

    // Calculate R (30) PP. 1145
    float R = sqrtf(Xi * Xi + Eta * Eta + q * q);

    // Calculate yTilde (30) PP. 1145
    float yTilde = Eta * cs + q * sn;

    // Calculate dTilde (30) PP. 1145
    float dTilde = Eta * sn - q * cs;

    // Calculate I1 - I5, (28) (29) PP. 1145
    // Order of calculation: I5, I4, I3 (uses I4), I2 (uses I3), I1 (Uses I5)
    // Each function checks for singularities and return 0 if calculation is not possible.

    float i5 = I5(Xi, Eta, Mu_L, R, dTilde, sn, cs, q, Eps);
    float i4 = I4(Xi, Eta, Mu_L, R, dTilde, sn, cs, q, Eps);
    float i3 = I3(Xi, Eta, Mu_L, R, yTilde, dTilde, sn, cs, q, i4, Eps);
    float i2 = I2(Xi, Eta, Mu_L, R, yTilde, dTilde, sn, cs, q, i3, Eps);
    float i1 = I1(Xi, Eta, Mu_L, R, dTilde, sn, cs, q, i5, Eps);

    // Fx, Fy, Fz: inner part of equation [ ... ]|| (26) PP. 1144 for ux, uy and uz

    // Initialize Fx, Fy, Fz to the last term of (26) PP. 1144 to return early
    // in case of zero or singularity
    // Fx : -I3 * sn * cs
    // Fy : -I1 * sn * cs
    // Fz : -I5 * sn * cs

    Fx = -i3 * sn * cs;
    Fy = -i1 * sn * cs;
    Fz = -i5 * sn * cs;

    // Terms are evaluated to zero or singularity exists:
    // If q is zero, first and second terms of Fx, Fy and Fz are set to zero:
    // On Fx, q sets the numerator to zero on the first term.
    // On Fy and Fz, q sets the numerator to zero on both first and second term.
    if (isZero(q, Eps))
    {
      return;
    }

    // Flags to check singularities on R and (R + Eta)
    bool zeroR = isZero(R, Eps);
    bool zeroRXi = isZero(R + Xi, Eps);

    if (!zeroR)
    {
      // Add first term of Fx
      Fx += q / R;
      if (!zeroRXi)
      {
        // Add first terms to Fy, Fz
        Fy += (yTilde * q) / (R * (R + Xi));
        Fz += (dTilde * q) / (R * (R + Xi));
      }

      // Add second terms to Fy, Fz
      Fy += cs * atanf((Xi * Eta) / (q * R));
      Fz += sn * atanf((Xi * Eta) / (q * R));
    }
  }

  float I1(
      float Xi,
      float Eta,
      float Mu_L,
      float R,
      float dTilde,
      float sn,
      float cs,
      float q,
      float i5,
      float Eps)
  {
    // I1 (28), (29), PP. 1144, 1145
    // Singularities when cs -> 0, (R + dTilde) -> 0

    // Set to zero cover base case, when cs -> 0 && (R + dTilde) -> 0
    float i1{0.0f};

    // Flags to check singularities
    bool zeroCs = isZero(cs, Eps);
    bool zeroRdTilde = isZero(R + dTilde, Eps);

    // Base case: cs -> 0 and (R + dTilde) -> 0
    if (zeroCs && zeroRdTilde)
    {
      return i1;
    }

    // POST: cs is not zero or (R + dTilde) is not zero

    // I5 factor only is calculated if cs is not zero
    float i5Factor = (zeroCs ? 0.0f : (sn / cs) * i5);

    // When cs -> 0, use  (29), otherwise use (28). (R + dTilde) -> 0 already covered by base case.
    i1 = (zeroCs ? -(Mu_L / 2.0f) * ((Xi * q) / ((R + dTilde) * (R + dTilde)))
                 : Mu_L * (-Xi / (cs * (R + dTilde))));

    // Substract i5Factor (or zero)
    i1 -= i5Factor;

    return i1;
  }

  float I2(
      float Xi,
      float Eta,
      float Mu_L,
      float R,
      float yTilde,
      float dTilde,
      float sn,
      float cs,
      float q,
      float i3,
      float Eps)
  {
    // I2 (28), (29) PP. 1144, 1145
    // Singularity when (R + Xi)

    // Set to zero
    float i2{0.0f};

    // (28) PP.1144. Replace ln(R + Eta) to -ln(R + Eta) when (R + Eta) -> 0
    i2 = Mu_L * ((R + Eta) >= Eps ? (-log(R + Eta)) : (log(R - Eta))) - i3;

    return i2;
  }

  float I3(
      float Xi,
      float Eta,
      float Mu_L,
      float R,
      float yTilde,
      float dTilde,
      float sn,
      float cs,
      float q,
      float i4,
      float Eps)
  {
    // I3 (28), (29) PP. 1144, 1145

    // Singularities when
    // cs -> 0, (R + dTilde) -> 0, (R + Eta) -> 0

    float i3{0.0f};

    bool zeroCs = isZero(cs, Eps);
    bool zeroRdTilde = isZero(R + dTilde, Eps);
    bool zeroREta = isZero(R + Eta, Eps);

    // When all singularities exist, return zero.
    if (zeroCs && zeroRdTilde && zeroREta)
    {
      return i3;
    }

    // If cs is zero, use (29) PP. 1145
    if (zeroCs)
    {
      // Check for singularities on (R + dTilde) and (R + Eta)

      // If (R + dTilde is not zero), add first two terms on I3 (29) PP. 1145
      if (!zeroRdTilde)
      {
        i3 += (Eta / (R + dTilde)) + ((yTilde * q) / ((R + dTilde) * (R + dTilde)));
      }

      // Substract third term on (29) PP. 1145
      // If (R + Eta) is not zero substract last term.
      if (!zeroREta)
      {
        i3 -= log(R + Eta);
      }
      else
      {
        // If (R + Eta) -> 0, replace ln(R + Eta) to - ln(R - Eta)
        // Substraction of a negative, change to addition
        i3 += log(R - Eta);
      }
      // Multiply bf Mu_l / 2
      i3 *= (Mu_L / 2.0f);
    }
    else
    {
      // cs is not zero, use (28) PP. 1144
      // Check for singularities on (R + dTilde) and (R + Eta)
      // cs -> 0 already covered on "if" block above.

      // Add (set) first term if there is no singularity
      if (!zeroRdTilde)
      {
        i3 += yTilde / (cs * (R + dTilde));
      }

      // Substract second term on (28) PP. 1144
      // If (R + Eta) substract last term.
      if (!zeroREta)
      {
        i3 -= log(R + Eta);
      }
      else
      {
        // If (R + Eta) -> 0, replace ln(R + Eta) to - ln(R - Eta)
        // Substraction of a negative, change to addition
        i3 += log(R - Eta);
      }

      // Multiply by Mu_L
      i3 *= Mu_L;

      // Add last term, cs -> 0 already covered!
      i3 += (sn / cs) * I4(Xi, Eta, Mu_L, R, dTilde, sn, cs, q, Eps);
    }

    return i3;
  }

  float I4(
      float Xi,
      float Eta,
      float Mu_L,
      float R,
      float dTilde,
      float sn,
      float cs,
      float q,
      float Eps)
  {
    // I4 (28), (29) PP. 1144, 1145
    // Singularities when cs -> 0, (R + dTilde) -> 0, (R + eta) -> 0

    // Set to zero
    float i4{0.0f};

    // Check flags for singularities
    bool zeroCs = isZero(cs, Eps);
    bool zeroRdTilde = isZero(R + dTilde, Eps);
    bool zeroREta = isZero(R + Eta, Eps);

    // Base case, cs -> 0 && (R + dTilde) -> 0 && (R + eta) -> 0
    if (zeroCs && zeroRdTilde && zeroREta)
    {
      return i4;
    }

    // If cs -> o, use (29) PP. 1145
    if (zeroCs)
    {
      // If (R + dTilde) is not zero, calculate (29) PP. 1114
      if (!zeroRdTilde)
      {
        i4 = -Mu_L * (q / (R + dTilde));
        return i4;
      }
    }
    else
    {
      // Use (28) PP. 1144

      // Add first term inside [ ... ] to i4
      // cs already checked
      if (!zeroRdTilde)
      {
        // if (R + dTilde) is not zero, add ln(R + dTilde)
        i4 += log(R + dTilde);
      }
      else
      {
        // (R + dTilde) -> 0, replace ln(R + dTilde) to -ln(R - dTilde)
        i4 -= log(R - dTilde);
      }

      // Substract second term inside [ ... ] to i4
      if (!zeroREta)
      {
        // If (R + Eta) is not zero, substract
        i4 -= sn * log(R + Eta);
      }
      else
      {
        // If (R + Eta) -> 0, replace ln(R + Eta) to -ln(R + Eta)
        // Addition, not substraction!
        i4 += sn * log(R - Eta);
      }

      // Multiply by Mu_L and 1/cs
      i4 *= Mu_L * (1.0f / cs);
    }

    return i4;
  }

  float I5(
      float Xi,
      float Eta,
      float Mu_L,
      float R,
      float dTilde,
      float sn,
      float cs,
      float q,
      float Eps)
  {
    // I5 (28), (29) PP. 1144, 1145

    // Set to zero to cover base cases:
    // Xi -> 0
    // Xi -> 0 && cs -> 0
    float i5{0.0f};

    // Check flags for singularities
    bool zeroXi = isZero(Xi, Eps);
    bool zeroCs = isZero(cs, Eps);
    bool zeroRdTilde = isZero(R + dTilde, Eps);

    // Base case, Xi -> 0, set I5 to zero. (PP. 1148), because:
    // Xi -> 0 causes singularity on I5 (28) and sets I5 (29) to zero.
    // Both Xi -> 0 and cs  -> 0 causes singularity on I5 (28) and sets I5 (29) to zero.
    if (zeroXi)
    {
      return i5;
    }

    // Check for singularity on cs
    if (zeroCs)
    {
      // cs -> 0, use (29) PP. 1145
      i5 = (zeroRdTilde ? 0.0
                        : -Mu_L * ((Xi * sn) / (R + dTilde)));
    }
    else
    {
      // cs is not zero, use (28) PP. 1144
      float X = sqrt(Xi * Xi + q * q);
      i5 = isZero(R + X, Eps) ? 0.0
                              : ((Mu_L * 2.0) / cs) * atanf(((Eta * (X + (q * cs))) + (X * (R + X) * sn)) / (Xi * (R + X) * cs));
    }

    return i5;
  }
} // End namespace
