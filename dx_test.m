clear all
close all
clc

% TWO STAGE PARTICLE FILTER__________________________________________

R = 8;
tic
parfor run = 1:R
    % Create data
    % Time series length
    T = 40;

    % State dimension
    dx = 400;
    dy = 400;
    dk = 10;

    % State, Observation, and Proposal noise
    var_x = 1;
    var_y = 1;
    var = 0.1;
    noise = {var_x, var_y, var};

    % State and observation range
    range = {[-1, 1, 0.3],[-2, 2, 0]} ;

    % State transition and observation functions
    g = @(x) 1./(1 + exp(-0.5*x));
    h = @(x) x;
    fns = {g, h};


    % Create data
    [x, y, C, H] = create_data(dx, dy, T, var_x, var_y, fns, range);

    coeffs = {C, H};

    % TPF settings
    M = 400;

    % Beta selection
    B = 0.1:0.1:1;

    % Run filters
    
    [x_grp, ~] = coupled_partition(y, coeffs, fns, noise, dk, M, B);
    [x_mpf] = twrp_mpf(y, M, var_x, var_y, g, C, H, dk);
   

    % Get MSE
    mse_grp(run) = sum(sum( (x_grp - x).^2 ))/(dx*T);
    mse_mpf(run) = sum(sum( (x_mpf - x).^2 ))/(dx*T);
end
toc

% mean(mse_grp,1)
% mean(mse_mpf,1)
% 
% plot(dks,mse_mpf)
% hold on
% plot(dks, mse_grp)

%load dx_dy60_dx60_R100.mat
%save('newdx_dy400_dx400_dk20_R100.mat')
