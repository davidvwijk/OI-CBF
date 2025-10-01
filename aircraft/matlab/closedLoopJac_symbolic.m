clear all; clc;

% Define states
syms phi theta psi pN pE H P Nz real
x = [phi; theta; psi; pN; pE; H; P; Nz];

% Define constants
syms up_min up_max uz_min uz_max g VT tp tz phimax kphi kP kN ktheta kH pN_star pE_star H_star real 

% Backup control
phi_d = phimax;
Nz_star = 1 / cos(phi_d);
up = P + tp*(kphi*(phi_d - phi) - kP*P);
uz = Nz + tz*(kN*(Nz_star - Nz) - ktheta*(theta) + kH*(H_star - H));

A = up_max;
s0_up = tanh(0.5 * 30 * A);
gamma_up = 1 / (s0_up + 1e-6);
up = -A ...
   + log(1 + exp(30 * (gamma_up*up + A))) / 30 ...
   - log(1 + exp(30 * (gamma_up*up - A))) / 30;
L = uz_min;
U = uz_max;
v0 = (1/30) * ( log(1 + exp(30*L)) - log(1 + exp(-30*U)) );
s0 = 1 / (1 + exp(-30*(0 - v0 - L))) ...
   - 1 / (1 + exp(-30*(0 - v0 - U)));
gamma = 1 / (s0 + 1e-6);
uz = L ...
   + log(1 + exp(30 * (gamma*uz - v0 - L))) / 30 ...
   - log(1 + exp(30 * (gamma*uz - v0 - U))) / 30;

% Closed-loop dynamics
fcl = [
    P + ((Nz*g)/VT)*sin(phi)*tan(theta);
    (g/VT)*(Nz*cos(phi) - cos(theta));
    (Nz*g*sin(phi))/(VT*cos(theta));
    VT*cos(theta)*cos(psi);
    VT*cos(theta)*sin(psi);
    VT*sin(theta);
    (1/tp)*(up-P);
    (1/tz)*(uz-Nz)
];

% Compute Jacobian
J = jacobian(fcl, x);

% Create function
matlabFunction(J, ...
    'Vars', {x, [up_min, up_max, uz_min, uz_max, g, VT, tp, tz, phimax, kphi, kP, kN, ktheta, kH, pN_star, pE_star, H_star]}, ...
    'File', 'jacobian_fun.m');
