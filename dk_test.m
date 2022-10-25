clear all
close all
clc

% TWO STAGE PARTICLE FILTER__________________________________________

% Load Data
load dx300_dy30_R100.mat

R = 100;

tic
parfor run = 1:R

    % Extract
    y = squeeze(y_store(run, :,:));
    x = squeeze(x_store(run, :,:));
    C = squeeze(C_store(run, :,:));
    H = squeeze(H_store(run, :,:));

    % Get time-series length
    T = length(y(1,:));
    dx = length(x(:,1));
    dy = length(y(:,1));

    % Beta selection
    B = 0.1:0.1:1;

    % State transition and observation functions
    g = @(x) 1./(1 + exp(-0.5*x));
    h = @(x) x;
    fns = {g, h};
    
    % State, Observation, and Proposal noise
    var_x = 1;
    var_y = 1;
    var = 0.1;
    noise = {var_x, var_y, var};
    coeffs = {C, H};
    
    % Number of particles
    M = 400;
    sizes = [2, 4, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50];
    sz = length(sizes);
    mse_grp = zeros(1,sz);
    mse_mpf = zeros(1,sz);


    tic
    for v = 1 : sz
        
        dk = sizes(v);
  
        [x_grp, b_grp] = coupled_partition(y, coeffs, fns, noise, dk, M, B);
        [x_mpf] = twrp_mpf(y, M, var_x, var_y, g, C, H, dk);

        mse_grp(v) = sum(sum( (x_grp - x).^2 ))/(dx*T);
        mse_mpf(v) = sum(sum( (x_mpf - x).^2 ))/(dx*T);

    end
    toc

    mse3(run, :) = mse_grp;
    mse5(run, :) = mse_mpf;


end
toc

clear all
clc

load dk_dx100_dy100res.mat
mse_grp = mean(mse3, 1);
%mse_mpf = mean(mse5,1);

%dk = [2, 4, 8, 10, 12, 16, 20, 25, 30];
%idx = [1, 2, 3, 5, 6];
%idx = 1:length(dk);
%save('dk_dx100_dy100res.mat')

idx = 1:length(dk);


plot(dk(idx), mse_grp(idx))
hold on
%plot(dk(idx), mse_mpf(idx))
legend('TPF')



