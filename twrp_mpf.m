function [x_est, x_samples] = twrp_mpf(y, M, var_x, var_y, g, C, H, dmax)


% Dimension
dy = length(y(:,1));
dx = length(C(1,:));

% Time series length
T = length(y(1,:));

x_old = mvnrnd(zeros(dx,1), eye(dx))';

A = (C~=0);

% MPF
for t = 2:T


    % Initialize new partitioning
    state_vector = 1:dx;
    k = 0;
    part = {};
    while (isempty(state_vector) == 0 )
        % Next partition
        k = k + 1;

        % If there is only one state left
        if (length(state_vector) <= dmax)
            part{k} = state_vector;
            state_vector = [];
        else

            % Choose one random state
            j = datasample(state_vector, 1);

            % Find ones from left over states
            idx_temp = find(A(:, j) == 1);
            [~, idx] = ismember(idx_temp, state_vector);
            idx = setdiff(idx, 0);
            idx1 = state_vector(idx);

            S = abs(C(idx1, j));
            wS = S/sum(S);

            % In case the left ones are 0s
            if (sum(S) == 0)
                part{k} = state_vector;
                state_vector = [];
            else


                % Sample from those that are ones (with replacement)
                links = datasample(idx1, min(dmax, length(idx1)), 'Weights', wS);

                % Form partition of chosen state j and links found
                part_idx = unique([j, links]);
                part{k} = part_idx;

                % Remove chosen indices from state_vector
                [~, idx] = ismember(part_idx, state_vector);
                state_vector(idx) =[];

            end


        end
    end


    % Number of partitions created at time t
    K = k;

    % Size of each partition
    dk = cell2mat(cellfun(@length, part,'uni',false));

    % Get proportionate number of particles for each filter
    Mk = floor( dk./dx * M);
    Mdiff  =  M - sum(Mk);
    idx = datasample(1:K, Mdiff, 'Replace', false);
    Mk(idx) = Mk(idx) + 1;


    % Filter l
    xk_temp ={};
    for k = 1 : K

        % Propose particles
        tr_mean = C(part{k}, :)*g(x_old);
        xk = mvnrnd( tr_mean' , var_x*eye(dk(k)), Mk(k))';

        % Store proposed
        xk_temp{k} = xk;

        % Get predictions
        xk_pred(part{k}, t) = mean(xk_temp{k}, 2);

    end

    w = [];
    % Compute weights
    for k = 1:K

        % Temp variable
        idk = setdiff(1:dx, part{k});
        xk_pred_temp = zeros(dx, Mk(k));
        for m = 1 : Mk(k)
            xk_pred_temp(part{k}, m) = xk_temp{k}(1:dk(k), m);
            xk_pred_temp(idk, m) = xk_pred(idk,t);
        end
        

        % Log weights of filter k
        log_wk = - 0.5*dk(k)*log(2*pi*var_y) - 0.5/var_y * sum( ( y(:,t) - H*xk_pred_temp ).^2 ,1 );

        % Rescale and normalize
        wk = exp(log_wk - max(log_wk));
        wk = wk./ sum(wk);

        % Resample
        idx = datasample(1:Mk(k), Mk(k), 'Weights', wk);

        % Get estimate
        xk = xk_temp{k};
        x_est(part{k},t) = mean( xk(:, idx), 2);

        % Store stream
        xk_store{t, k} = xk(:,idx); 

        w = [w, wk];


    end


    x_old = x_est(:,t);


end

% sample
x_samples = mvnrnd(x_est, 0.1*eye(T));



end