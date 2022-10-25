clear all
close all
clc

% TWO STAGE PARTICLE FILTER__________________________________________

% Create data
% Time series length
T = 40;

% State dimension
dx = 100;
dy = 100;


% State, Observation, and Proposal noise
var_x = 0.1;
var_y = 1;
var = 0.1;
noise = {var_x, var_y, var};

% State and observation range
range{1} = [-1, 1, 0.3];
range{2} = [-2, 2, 0];

% State transition and observation functions
g = @(x) 1./(1 + exp(-1*x));
%h = @(x) exp(-0.5*x);
h = @(x) x;
fns = {g, h};


% Create data
[x, y, C, H] = create_data(dx, dy, T, var_x, var_y, fns, range);

coeffs = {C, H};

% TPF settings
M = 50;

% Beta selection
B = 0.2:0.05:0.4;


tic
dk = 1;
R = 3;
[x_grp, b_grp] = coupled_partition(y, coeffs, fns, noise, dk, M, B, R);
toc

tic
dk = 10;
[x_mpf] = twrp_mpf(y, M, var_x, var_y, g, C, H, dk);
toc


mse_grp = sum(sum( (x_grp - x).^2 ))/(dx*T)
mse_mpf = sum(sum( (x_mpf - x).^2 ))/(dx*T)

close all
% Plot trajectory of random state
j = datasample(1:dx, 1);
t0 = 1; tT = 40;

figure(4)
plot(x(j,t0:tT), 'k', 'LineWidth',2)
hold on
plot(x_grp(j,t0:tT), 'm', 'LineStyle','--', 'LineWidth',1.5)
hold on
plot(x_mpf(j,t0:tT), 'p', 'LineStyle','--', 'LineWidth',1.5)
xlabel('Time', 'FontSize',20)
ylabel('State', 'FontSize',20)
legend('True State', 'TPF', 'MPF', 'FontSize', 20)

%save('dy100_dx100_M50track.mat', 'x', 'x_mpf', 'x_grp')





