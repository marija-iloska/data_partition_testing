function [x, y, C, H] = create_data(dx, dy, T, var_x, var_y, fns, range)

% State transition and observation functions
g = fns{1};     h = fns{2};

% Initialize arrays for true state x and observation y
x = zeros(dx, T);
y = zeros(dy, T);
x(:,1) = rand(dx,1);

% State coefficient
C = unifrnd(range{1}(1), range{1}(2), dx, dx);
H = unifrnd(range{2}(1), range{2}(2), dy, dx);

% Some zeros (some degree of separability)
for j = 1:dx
    idx = setdiff(datasample(1:dx, round(range{1}(3)*dx)), j);
    C(j,idx) = 0;
end

for j=1:dy
    idx = setdiff(datasample(1:dx, round(range{2}(3)*dx)), j);
    H(j,idx) = 0;
end


% Time series data
for t = 2:T
    x(:,t) = C*g(x(:,t-1)) + mvnrnd(zeros(1,dx), var_x*eye(dx))';
    y(:,t) = H*h(x(:,t)) + mvnrnd(zeros(1,dy), var_y*eye(dy))';
end


end