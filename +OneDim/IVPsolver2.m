classdef IVPsolver2 < handle
    % IVPSOLVER2 Solver for learning sub- and super-solutions
    %
    % This class trains two neural networks (v and w) to learn error bounds
    % such that u-v is a sub-solution and u+w is a super-solution.
    
    properties
        initialCondition
        odeFunction
        tspan
        optimizer
        hfunction
        error_tolerances
        loss_coefficients
        penalty_factor = 1e2
        LossFunction
        model_app = []  % Pre-trained approximate solution (required)
    end

    methods
        function obj = IVPsolver2(initialCondition, odeFunction, tspan, varargin)
            % IVPSOLVER2 Constructor
            %
            % Inputs:
            %   initialCondition - Initial value u(0)
            %   odeFunction - ODE function handle @(u, t)
            %   tspan - Time domain [t_start, t_end]
            %
            % Optional Name-Value pairs:
            %   'error_tolerances' - Error tolerance epsilon (default: 0.1)
            %   'loss_coefficients' - Loss weights [1,1,1,1,1]
            %   'penalty_factor' - Penalty on df/du (default: 1e2)
            %   'model_app' - Pre-trained approximate solution (OneDim.NN)
            %   'LossFunction' - Custom loss function handle
            
            obj.initialCondition = initialCondition;
            obj.odeFunction = odeFunction;
            obj.tspan = tspan;
            
            % Default values
            obj.error_tolerances = 0.1; 
            obj.loss_coefficients = [1, 1, 1, 1, 1]; 
            obj.hfunction = @(t, epsilon) abs(t-epsilon);
            obj.LossFunction = @OneDim.IVPsolver2.customLossFunction;

            for i = 1:2:length(varargin)
                if strcmp(varargin{i}, 'error_tolerances')
                    obj.error_tolerances = varargin{i + 1};
                elseif strcmp(varargin{i}, 'loss_coefficients')
                    obj.loss_coefficients = varargin{i + 1};
                elseif strcmp(varargin{i}, 'penalty_factor')
                    obj.penalty_factor = varargin{i + 1};
                elseif strcmp(varargin{i}, 'lossfunc')
                    obj.LossFunction = varargin{i + 1};
                elseif strcmp(varargin{i}, 'model_app')
                    obj.model_app = varargin{i + 1};
                else
                    obj.(varargin{i}) = varargin{i + 1};
                end
            end

            model = OneDim.NN( ...
                'layerNum', 5, ...
                'neuronNumParOnelayer', 30, ...
                'domainStart', tspan(1), 'domainEnd', tspan(2), ...
                'initializationMethod','He', ...
                'dataType', 'double'...
                );

            obj.optimizer = OneDim.Optimizer({model,model}, 'auto', @obj.lossfuncWrapper, ...
                'numEpoch', 100, ...
                'numIterPerEpoch', 20, ...
                'batchSize', 256, ...
                'learnRate', 0.01, ...
                'convergenceThreshold', 1e-6, ...
                'earlyStopThreshold', 1e-8, ...
                'computeMode', 'cpu');
        end

        function [loss1, loss2] = lossfuncWrapper(obj, model1, model2, t)
            args = {model1, model2, t, obj.initialCondition, obj.odeFunction};
            
            if ~isempty(obj.penalty_factor)
                args = [args, {'penalty_factor', obj.penalty_factor}];
            end
            if ~isempty(obj.model_app)
                args = [args, {'model_app', obj.model_app}];
            end
            
            [loss1, loss2] = obj.LossFunction(args{:});
        end

        function obj = solve(obj)
            % SOLVE Train the two neural networks
            
            % Transfer model_app parameters to GPU if needed
            isGpuMode = strcmp(obj.optimizer.computeMode, 'gpu');
            
            if isGpuMode && ~isempty(obj.model_app)
                % Check if parameters are already on GPU to avoid double conversion
                firstW = extractdata(obj.model_app.parameters{1}.W);
                if ~isa(firstW, 'gpuArray')
                    params = cell(size(obj.model_app.parameters));
                    for k = 1:length(obj.model_app.parameters)
                        w_data = extractdata(obj.model_app.parameters{k}.W);
                        b_data = extractdata(obj.model_app.parameters{k}.b);
                        params{k}.W = dlarray(gpuArray(w_data));
                        params{k}.b = dlarray(gpuArray(b_data));
                    end
                    obj.model_app.parameters = params;
                end
            end
            
            obj.optimizer = obj.optimizer.trainTwoModels();
            
            % Transfer model_app parameters back to CPU
            if isGpuMode && ~isempty(obj.model_app)
                 params = cell(size(obj.model_app.parameters));
                 for k = 1:length(obj.model_app.parameters)
                    w_data = extractdata(obj.model_app.parameters{k}.W);
                    b_data = extractdata(obj.model_app.parameters{k}.b);
                    params{k}.W = dlarray(gather(w_data));
                    params{k}.b = dlarray(gather(b_data));
                 end
                 obj.model_app.parameters = params;
            end
        end
    end

    methods (Static)
        function [loss1, loss2] = customLossFunction(model1, model2, t, initialCondition, odeFunction, varargin)
            % CUSTOMLOSSFUNCTION Loss function for sub-/super-solution learning
            %
            % Uses Doubly Smoothed Maximum (DSM) to penalize constraint violations.
            %
            % Required Name-Value pairs:
            %   'model_app' - Pre-trained approximate solution (OneDim.NN)

            p = inputParser;
            addParameter(p, 'model_app', []);
            addParameter(p, 'penalty_factor', 1e2);
            parse(p, varargin{:});
            model_app = p.Results.model_app;
            
            if isempty(model_app)
                error('IVPsolver2:missingModelApp', ...
                    'model_app is required. Provide a pre-trained OneDim.NN model.');
            end
            penalty_factor = p.Results.penalty_factor;

            [v, dv] = model1.d_ann(t);  % v >= 0 (sub-solution error)
            [w, dw] = model2.d_ann(t);  % w >= 0 (super-solution error)
            [u, du] = model_app.d_ann(t);

            subsol = u - v;
            supsol = u + w;
            dsubsol = du - dv;
            dsupsol = du + dw;

            % DSM parameters
            c1 = 1e-2;
            c2 = 1e-3;
            
            % Doubly Smoothed Maximum function
            DSM = @(t,M) M + c2*log(sum((exp(-M/c1) + exp((t - M)/c1)).^(c1/c2)));

            % Sub-solution constraint: d(u-v)/dt <= f(u-v, t)
            g_sub = dsubsol - odeFunction(subsol, t);
            M_sub = max(max(g_sub, 0));
            penalty_subsol_eq = DSM(g_sub, M_sub);
            
            % Super-solution constraint: d(u+w)/dt >= f(u+w, t)
            g_sup = odeFunction(supsol, t) - dsupsol;
            M_sup = max(max(g_sup, 0));
            penalty_supsol_eq = DSM(g_sup, M_sup);

            if penalty_factor ~= 0
                delta = 1e-10;
                dfdsubsol = (odeFunction(subsol + delta, t) - odeFunction(subsol, t)) / delta;
                dfdsupsol = (odeFunction(supsol + delta, t) - odeFunction(supsol, t)) / delta;

                penalty_subsol = max(0, mean(dfdsubsol));
                penalty_supsol = max(0, mean(dfdsupsol));

                loss1 = penalty_subsol_eq + penalty_factor * penalty_subsol;
                loss2 = penalty_supsol_eq + penalty_factor * penalty_supsol;
            else
                loss1 = penalty_subsol_eq;
                loss2 = penalty_supsol_eq;
            end                
        end
    end
end
