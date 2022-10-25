clear all
close all
clc


R = 100;

parfor run = 1:R

    % All 3 for comparison
    %% Generate Data

    T = 40;

    % State dimension
    dx = 300;
    dy = 300;

    % State, Observation, and Proposal noise
    var_x = 1;
    var_y = 1;

    % State and observation range
    range = {[-2, 2, 0.3],[-2, 2, 0]};

    % State transition and observation functions
    g = @(x) 1./(1 + exp(-0.5*x));
    h = @(x) x;
    fns = {g, h};

    % Create data
    [x, y, C, H] = create_data(dx, dy, T, var_x, var_y, fns, range);


    % Store
    y_store(run, :,:) = y;
    x_store(run, :,:) = x;
    C_store(run, :,:) = C;
    H_store(run, :,:) = H;

end

% State dimension
dx = 100;
dy = 300;

save('dx300_dy300_R100.mat', 'y_store', 'x_store','C_store','H_store', "dx","dy");

