clear all
close all
clc

% TWO STAGE PARTICLE FILTER__________________________________________

R = 1;
tic
for run = 1:R
    % Create data
    % Time series length
    T = 40;

    % State dimension
    dx = 400;
    dy = 100;
    dk = 10;

    % State, Observation, and Proposal noise
    var_x = 1;
    var_y = 0.1;
    var = 0.5;
    noise = {var_x, var_y, var};

    % State and observation range
    range = {[-1, 1, 0.3],[-2, 2, 0.3]} ;

    % State transition and observation functions
    g = @(x) 1./(1 + exp(-x));
    h = @(x) x;
    fns = {g, h};


    % Create data
    [x, y, C, H] = create_data(dx, dy, T, var_x, var_y, fns, range);

    coeffs = {C, H};

    % TPF settings
    M = 500;

    % Beta selection
    B = 0.1:0.1:1;

    % Run filters
    
    [x_grp, ~] = coupled_partition(y, coeffs, fns, noise, dk, M, B, 4);
    [x_mpf] = twrp_mpf(y, M, var_x, var_y, g, C, H, dk);
   

    % Get MSE
    mse_grp(run) = sum(sum( (x_grp - x).^2 ))/(dx*T);
    mse_mpf(run) = sum(sum( (x_mpf - x).^2 ))/(dx*T);
end
toc

mean(mse_grp,1)
mean(mse_mpf,1)

mg = [237, 110, 152]/256;
%bg = [122, 186, 126]/256;

bg = [39, 163, 151]/256;

t0 = 10;
tT = 40;
j = datasample(1:dx, 1);
plot(x(j,t0:tT), 'k', 'LineWidth',2)
hold on
plot(x_grp(j,t0:tT), 'Color', 'm', 'LineStyle','--', 'LineWidth', 2)
hold on
plot(x_mpf(j,t0:tT), 'Color', bg, 'LineStyle','-.', 'LineWidth', 2)
set(gca, 'FontSize',15)
xlabel('Time', 'FontSize',20)
ylabel('State', 'FontSize',20)
legend('True State', 'RP-TPF', 'TWRP-MPF', 'FontSize', 20)
% 
% plot(dks,mse_mpf)
% hold on
% plot(dks, mse_grp)

%load dx_dy60_dx60_R100.mat
%save('newdx_dy400_dx400_dk20_R100.mat')
