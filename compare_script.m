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
dk = 10;

% State, Observation, and Proposal noise
var_x = 1;
var_y = 0.1;
var = 1;
noise = {var_x, var_y, var};

% State and observation range
range{1} = [-2, 2, 0.3];
range{2} = [-2, 2, 0];

% State transition and observation functions
g = @(x) 1./(1 + exp(-0.5*x));
%h = @(x) exp(-0.5*x);
h = @(x) x;
fns = {g, h};

% Initialize arrays for true state x and observation y
x = zeros(dx, T);
y = zeros(dy, T);
x(:,1) = rand(dx,1);

% Create data
[x, y, C, H] = create_data(dx, dy, T, var_x, var_y, fns, range);

coeffs = {C, H};

% TPF settings
M = 500;

% Beta selection
B = 0.1:0.1:1;
%B = 0.2:0.2:1;


[x_top, b_top] = topology_partition(y, coeffs, fns, noise, M, B);
[x_rnd, b_rnd] = random_partition(y, coeffs, fns, noise, M, B);
[x_grp, b_grp] = coupled_partition(y, coeffs, fns, noise, dk, M, B);
[x_tpc, b_tpc] = topology_coupled_partition(y, coeffs, fns, noise, dk, M,B);


mse_top = sum(sum( (x_top- x).^2 ))/(dx*T)
mse_rnd = sum(sum( (x_rnd - x).^2 ))/(dx*T)
mse_grp = sum(sum( (x_grp - x).^2 ))/(dx*T)
mse_tpc = sum(sum( (x_tpc - x).^2 ))/(dx*T)

close all
% Plot trajectory of random state
j = datasample(1:dx, 1);
t0 = 1; tT = 40;
figure(1)
plot(x(j,t0:tT), 'k', 'LineWidth',2)
hold on
plot(x_top(j,t0:tT), 'r', 'LineStyle','--', 'LineWidth',1.5)
legend('True State', 'Topology single', 'FontSize', 20)

figure(2)
plot(x(j,t0:tT), 'k', 'LineWidth',2)
hold on
plot(x_rnd(j,t0:tT), 'b', 'LineStyle','--', 'LineWidth',1.5)
xlabel('Time', 'FontSize',20)
ylabel('State', 'FontSize',20)
legend('True State', 'Random single', 'FontSize', 20)

figure(3)
plot(x(j,t0:tT), 'k', 'LineWidth',2)
hold on
plot(x_tpc(j,t0:tT), 'g', 'LineStyle','--', 'LineWidth',1.5)
xlabel('Time', 'FontSize',20)
ylabel('State', 'FontSize',20)
legend('True State', 'Topology Coupled', 'FontSize', 20)

figure(4)
plot(x(j,t0:tT), 'k', 'LineWidth',2)
hold on
plot(x_grp(j,t0:tT), 'm', 'LineStyle','--', 'LineWidth',1.5)
xlabel('Time', 'FontSize',20)
ylabel('State', 'FontSize',20)
legend('True State', 'Random Coupled', 'FontSize', 20)

% 
% 
% figure(4)
% hist(b_tpc)
% 
% figure(5)
% hist(b_rnd)
% 
% figure(6)
% hist(b_grp)
% 
% figure(7)
% hist(b_top)



