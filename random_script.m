clear all
close all
clc

% TWO STAGE PARTICLE FILTER__________________________________________
% SSM:
% x(t) = C*g( x(t-1) ) + Gaussian noise
% y(t) = exp( x(t)/2 ) + Gaussian noise


% Create data
% Time series length
T = 40;

% State dimension
dx = 20;
dy = 20;

% State, Observation, and Proposal noise
var_x = 1;
var_y = 0.1;
var = 1;

% State and observation coeff
C = unifrnd(-0.1, 0.1, dx,dx);
H = unifrnd(-2, 2, dy, dx);

for j = 1:dx
    idx = setdiff(datasample(1:dx, round(0.3*dx)), j);
    C(j,idx) = 0;
end

for j=1:dy
    idx = setdiff(datasample(1:dx, round(0.4*dx)), j);
    H(j,idx) = 0;
end

% State transition and observation functions
g = @(x) 1./(1 + exp(-x));
h = @(x) exp(-0.5*x);

% Initialize arrays for true state x and observation y
x = zeros(dx, T);
y = zeros(dy, T);
x(:,1) = rand(dx,1);

% Time series data
for t = 2:T
    x(:,t) = C*g(x(:,t-1)) + mvnrnd(zeros(1,dx), var_x*eye(dx))';
    y(:,t) = H*h(x(:,t)) + mvnrnd(zeros(1,dy), var_y*eye(dy))';
end

% TPF settings
M = 300;
beta = 0.2;

% Initialize 
x_particles = rand(dx,M);
x_old = x_particles;
x_est = zeros(dx, T);

% Beta selection
B = 0.05:0.05:1;
B = 0.2:0.2:1;
b_size = length(B);
beta_post = ones(1,b_size)./b_size;
w = ones(1,M)/M;
choice = zeros(1,T);
choice(1) = beta;

for t=2:T

    % FIRST STAGE
    % Propose iniital particles based on model transition
    for m = 1:M
        tr_mean(:,m) = C*g(x_old(:,m));
        x_particles(:,m) = mvnrnd(tr_mean(:,m), var_x*eye(dx))';
    end

    x_predicted = mean(x_particles,2);
    states_idx = 1:dx;

    % Modify proposal
    for j = 1:dy
        % Choose a random dimension
        k = datasample(states_idx, 1);

        for m = 1:M
            x_predicted(k) = x_particles(k,m);
            % Compute likelihood
            % - 0.5*log(2*pi*var_y)
            ln_p(m) = - (0.5/var_y)*(y(j,t) - H(j,:)*h(x_predicted) ).^2;
        end 
        p = exp(ln_p - max(ln_p));

        % Find max
        if (length(find(p == max(p))) ~= 1)
            m_star(j) = datasample(1:M, 1);
        else
            m_star(j) = find(p == max(p));
        end

        % Form proposed mean from particles with ML
        x_predicted(k) = x_particles(k, m_star(j));
        states_idx = setdiff(states_idx, k);
    end   

    % Beta posterior computation
    for b = 1:b_size
        beta = B(b);
        new_mean = beta*x_predicted + (1 - beta)*tr_mean;
        new_var = beta^2*var_x + (1- beta)^2*var;
        x_particles = mvnrnd(new_mean', new_var*eye(dx))';
        ln_l = - (0.5/var_y)*sum( (y(:,t) - H*h(x_particles) ).^2, 1 );
        l = exp(ln_l - max(ln_l));
        beta_post(b) =  beta_post(b)*sum(l.*w); 
    end
    beta_post = beta_post./sum(beta_post);
    beta = datasample(B, 1, 'Weights',beta_post);
    choice(t) = beta;

    % SECOND STAGE
    % Propose new particles
    for m = 1:M
        % New mean
        new_mean = beta*x_predicted + (1 - beta)*tr_mean(:,m);
        new_var = beta^2*var_x + (1- beta)^2*var;
        x_particles(:,m) = mvnrnd(new_mean, new_var*eye(dx))';

        ln_l(m) = - 0.5*dx*log(2*pi*var_y) - (0.5/var_y)*sum( (y(:,t) - H*h(x_particles(:,m)) ).^2 ) ;
        ln_t(m) = - 0.5*dx*log(2*pi*var_x) - (0.5/var_x)*sum( (x_particles(:,m) - tr_mean(:,m) ).^2 ) ;
        ln_q(m) = - 0.5*dx*log(2*pi*new_var) - (0.5/new_var)*sum( (x_particles(:,m) - new_mean ).^2 ) ;  

        ln_w(m) = ln_l(m) + ln_t(m) - ln_q(m);
    end

    w = exp(ln_w - max(ln_w));
    w = w./sum(w);


    % Resample
    idx = datasample(1:M, M, 'Weights',w);

    % Set new particles
    x_particles = x_particles(:,idx);
    x_old = x_particles;
    x_est(:,t) = mean(x_particles,2);


end

% Plot trajectory of random state
j = datasample(1:dx, 1);
plot(x(j,:), 'k', 'LineWidth',1.5)
hold on
plot(x_est(j,:), 'r', 'LineStyle','--', 'LineWidth',1)
xlabel('Time', 'FontSize',20)
ylabel('State', 'FontSize',20)
legend('True State', 'Estimate', 'FontSize', 20)











