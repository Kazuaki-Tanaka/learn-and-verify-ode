%% Simple Logistic Equation Example
% 
% This is a minimal example demonstrating the Learn and Verify approach
% for the classical logistic equation.
%
% ODE: du/dt = r*u*(1 - u/k)
%
% Parameters: T=10, r=1, k=2, a=0.5
%
% Estimated runtime: 2-3 minutes
%
% Author: Kazuaki Tanaka
% Last Updated: January 2026

clear; close all;
import OneDim.*

fprintf('========================================\n');
fprintf('Simple Logistic Equation Example\n');
fprintf('========================================\n\n');

%% Problem Setup

% ODE: du/dt = ru(1 - u/k)
r = 1;   % Growth rate
k = 2;   % Carrying capacity
T = 10;  % Time domain end
a = 0.5; % Initial condition

odeFunction = @(u, t) r .* u .* (1 - u ./ k);

% Initial condition and time domain
initialCondition = a;
tspan = [0, T];

% Analytical solution: u(t) = k / (1 + ((k/a) - 1) * exp(-r*t))
trueSolution = @(t) k ./ (1 + ((k/a) - 1) * exp(-r*t));

fprintf('Problem:\n');
fprintf('  ODE: du/dt = r*u*(1 - u/k)\n');
fprintf('  Parameters: T=%d, r=%d, k=%d, a=%.1f\n', T, r, k, a);
fprintf('  Analytical solution: u(t) = k / (1 + ((k/a)-1)*exp(-rt))\n\n');

%% Step 1: Learn Approximate Solution

fprintf('Step 1: Learning approximate solution...\n');

% Create solver with penalty factor
solver = IVPsolver(initialCondition, odeFunction, tspan, ...
    'penalty_factor', 2^-4);

% Configure neural network
solver.optimizer.model = NN( ...
    'layerNum', 5, ...                    % 4 hidden + 1 output
    'neuronNumParOnelayer', 30, ...       % 30 neurons per layer
    'domainStart', tspan(1), ...
    'domainEnd', tspan(2), ...
    'initializationMethod', 'SIREN');

% Scale first layer weights
solver.optimizer.model.parameters{1}.W = ...
    solver.optimizer.model.parameters{1}.W * 50;

% Training settings
solver.optimizer.numEpoch = 50;
solver.optimizer.numIterPerEpoch = 50;
solver.optimizer.batchSize = 128;
solver.optimizer.learnRate = 0.01;
solver.optimizer.earlyStopThreshold = 0;
solver.optimizer.regionCount = 10;
solver.optimizer.trainingMode = 'region';

% Train
tic;
solver = solver.solve();
training_time_approx = toc;

% Adjust initial condition exactly
solver.optimizer.model.parameters{end}.b = ...
    solver.optimizer.model.parameters{end}.b + ...
    initialCondition - solver.optimizer.model.ann(dlarray(tspan(1)));

% Evaluate approximation quality
t_eval = linspace(tspan(1), tspan(2), 500);
u_approx_eval = extractdata(solver.optimizer.model.ann(dlarray(t_eval)));
u_true_eval = trueSolution(t_eval);
approxError = max(abs(u_approx_eval - u_true_eval));

fprintf('  Training time: %.2f seconds\n', training_time_approx);
fprintf('  Max approximation error: %.4e\n', approxError);
fprintf('  Initial condition: u(0) = %.10f (target: %.1f)\n\n', ...
    extractdata(solver.optimizer.model.ann(dlarray(tspan(1)))), initialCondition);

%% Step 2: Learn Sub- and Super-solutions

fprintf('Step 2: Learning sub- and super-solutions...\n');

% Error tolerance
epsilon = 2^-4;
fprintf('  Error tolerance: epsilon = 2^-4 = %.4f\n', epsilon);

% Create solver for sub-/super-solutions
solver2 = IVPsolver2(initialCondition, odeFunction, tspan, ...
    'error_tolerances', epsilon, ...
    'penalty_factor', 0, ...
    'model_app', solver.optimizer.model, ...    
    'LossFunction', @OneDim.IVPsolver2.customLossFunction);

% Configure networks
for i = 1:2
    solver2.optimizer.model{i} = NN( ...
        'layerNum', 4, ...                    % 3 hidden + 1 output
        'neuronNumParOnelayer', 30, ...
        'domainStart', tspan(1), ...
        'domainEnd', tspan(2), ...
        'initializationMethod', 'SIREN');
    
    % Scale first layer
    solver2.optimizer.model{i}.parameters{1}.W = ...
        solver2.optimizer.model{i}.parameters{1}.W * 100;
end

% Sigmoid constraint: v, w in (0, epsilon)
mysigmoid = struct(...
    'f', @(x) epsilon ./ (1 + exp(-x)), ...
    'df', @(x) (epsilon .* exp(-x)) ./ (exp(-x) + 1).^2, ...
    'ddf', @(x) -(epsilon .* exp(x) .* (exp(x) - 1)) ./ (exp(x) + 1).^3, ...
    'dddf', @(x) (epsilon .* exp(x) .* (exp(2 .* x) - 4 .* exp(x) + 1)) ./ (exp(x) + 1).^4, ...
    'ddddf', @(x) -(epsilon .* exp(x) .* (11 .* exp(x) - 11 .* exp(2 .* x) + exp(3 .* x) - 1)) ./ (exp(x) + 1).^5);

solver2.optimizer.model{1}.lastLayerActivation = mysigmoid;
solver2.optimizer.model{2}.lastLayerActivation = mysigmoid;

% Training settings
solver2.optimizer.numEpoch = 100;
solver2.optimizer.numIterPerEpoch = 50;
solver2.optimizer.learnRate = 1e-3;
solver2.optimizer.batchSize = 512;
solver2.optimizer.earlyStopThreshold = 0;
solver2.optimizer.regionCount = 20;
solver2.optimizer.trainingMode = 'region';

% Train
tic;
solver2 = solver2.solve();
training_time_subsup = toc;

fprintf('  Training time: %.2f seconds\n\n', training_time_subsup);

%% Step 3: Rigorous Verification

fprintf('Step 3: Rigorous verification with interval arithmetic...\n');

% Verify using INTLAB
tic;
result = RigorousNN.odeVerifyer(...
    solver2.optimizer.model{1}, ...     % v model (sub-solution error)
    solver2.optimizer.model{2}, ...     % w model (super-solution error)
    solver.optimizer.model, ...         % approximate solution
    odeFunction, ...
    initialCondition, ...
    100);                               % Number of initial intervals
verification_time = toc;

fprintf('  Verification time: %.2f seconds\n', verification_time);

if result.verificationSuccess
    fprintf('\n  ✓ SUCCESS! Rigorous enclosure verified.\n');
    fprintf('    Error bound (w+v): %.6e\n', result.wvUpperBound);
    fprintf('    Sub-solution margin: %.6e\n', result.marginSubsol);
    fprintf('    Super-solution margin: %.6e\n', result.marginSupsol);
else
    fprintf('\n  ✗ FAILED: %s\n', result.failReason);
    fprintf('    This may happen with insufficient training.\n');
    fprintf('    Try: increase epochs or use more neurons.\n');
end

%% Step 4: Visualization

fprintf('\nStep 4: Visualizing results...\n');

% Generate plot points
t_plot = linspace(tspan(1), tspan(2), 200);

% Evaluate models
u_approx = extractdata(solver.optimizer.model.ann(dlarray(t_plot)));
v = extractdata(solver2.optimizer.model{1}.ann(dlarray(t_plot)));
w = extractdata(solver2.optimizer.model{2}.ann(dlarray(t_plot)));

% Analytical solution
u_true = trueSolution(t_plot);

% Compute sub- and super-solutions
u_sub = u_approx - v;
u_sup = u_approx + w;

% Create figure
figure('Position', [100, 100, 1200, 400]);

% Subplot 1: Solution enclosure
subplot(1, 2, 1);
fill([t_plot, fliplr(t_plot)], [u_sub, fliplr(u_sup)], ...
     [0.9, 0.95, 1], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
hold on;
plot(t_plot, u_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True solution');
plot(t_plot, u_approx, 'g--', 'LineWidth', 1.5, 'DisplayName', 'Approximate');
plot(t_plot, u_sub, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Sub-solution');
plot(t_plot, u_sup, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Super-solution');
hold off;
grid on; box on;
xlabel('Time t', 'FontSize', 12);
ylabel('u(t)', 'FontSize', 12);
title('Solution Enclosure', 'FontSize', 14);
legend('Location', 'southeast', 'FontSize', 9);

% Subplot 2: Error bounds (v and w)
subplot(1, 2, 2);
plot(t_plot, v, 'b-', 'LineWidth', 2, 'DisplayName', 'v (sub-error)');
hold on;
plot(t_plot, w, 'r-', 'LineWidth', 2, 'DisplayName', 'w (sup-error)');
plot(t_plot, v + w, 'k-', 'LineWidth', 2, 'DisplayName', 'v + w');
yline(epsilon, 'k--', 'LineWidth', 1.5, 'DisplayName', '\epsilon');
hold off;
grid on; box on;
xlabel('Time t', 'FontSize', 12);
ylabel('Error', 'FontSize', 12);
title('Error Bounds', 'FontSize', 14);
legend('Location', 'northeast', 'FontSize', 9);


sgtitle(sprintf('Classical Logistic: r=%d, k=%d, a=%.1f, T=%d', r, k, a, T), ...
    'FontSize', 16, 'FontWeight', 'bold');

%% Summary

fprintf('\n========================================\n');
fprintf('Summary\n');
fprintf('========================================\n');
fprintf('Total time: %.2f seconds\n', training_time_approx + training_time_subsup + verification_time);
fprintf('  - Approximate solution: %.2f s\n', training_time_approx);
fprintf('  - Sub-/super-solutions: %.2f s\n', training_time_subsup);
fprintf('  - Verification: %.2f s\n', verification_time);
fprintf('\n');

if result.verificationSuccess
    fprintf('✓ The true solution u(t) is rigorously guaranteed to lie\n');
    fprintf('  between u_sub(t) and u_sup(t) for all t in [0, %d].\n', T);
    fprintf('\n');
    fprintf('Enclosure width (max of w+v): %.6e\n', result.wvUpperBound);
    fprintf('  (The true solution lies within this width of the approximate solution)\n');
    fprintf('Relative enclosure width: %.4f%% of max(u)\n', result.wvUpperBound/max(u_true)*100);

    % Check Global Existence (Theorem 3 in the paper)
    % Verify if the solution can be extended to t -> infinity
    u_vmodel = OneDim.RigorousNN(solver.optimizer.model);
    v_vmodel = OneDim.RigorousNN(solver2.optimizer.model{1});
    w_vmodel = OneDim.RigorousNN(solver2.optimizer.model{2});
    
    % Evaluate at T
    u_end = u_vmodel.ann(T);
    v_end = v_vmodel.ann(T);
    w_end = w_vmodel.ann(T);
    
    subsol_end = u_end - v_end;
    supsol_end = u_end + w_end;
    
    % Check differential inequalities for constant extension:
    % sub_sol' <= f(sub_sol)  =>  0 <= f(sub_sol_end)
    % sup_sol' >= f(sup_sol)  =>  0 >= f(sup_sol_end)
    f_sub = odeFunction(subsol_end, T);
    f_sup = odeFunction(supsol_end, T);
    
    if f_sub.inf >= 0 && f_sup.sup <= 0
        fprintf('\n  ✓ GLOBAL EXISTENCE VERIFIED!\n');
        fprintf('    The solution exists for all t >= 0 and remains within\n');
        fprintf('    [%.6f, %.6f] for t >= %d.\n', subsol_end.inf, supsol_end.sup, T);
    else
        fprintf('\n  - Global existence check failed at t=%d.\n', T);
        fprintf('    (This is optional and does not invalidate the enclosure on [0, %d])\n', T);
    end
else
    fprintf('✗ Verification failed. Consider:\n');
    fprintf('  - Training longer (increase numEpoch)\n');
    fprintf('  - Using tighter error tolerance (smaller epsilon)\n');
    fprintf('  - Increasing network size (more neurons or layers)\n');
end

fprintf('\n========================================\n');
fprintf('Example complete!\n');
fprintf('========================================\n');
