classdef IVPsolver < handle
    % IVPSOLVER Solver for learning approximate ODE solutions using neural networks
    %
    % This class trains a neural network to approximate the solution of an
    % initial value problem (IVP) using a physics-informed loss function.
    
    properties
        initialCondition
        odeFunction
        tspan
        optimizer
        penalty_factor = 1e2;
        initialConditionWeight = 1;
    end

    methods
        function obj = IVPsolver(initialCondition, odeFunction, tspan, varargin)
            % IVPSOLVER Constructor
            %
            % Inputs:
            %   initialCondition - Initial value u(0)
            %   odeFunction - ODE function handle @(u, t)
            %   tspan - Time domain [t_start, t_end]
            %
            % Optional Name-Value pairs:
            %   'penalty_factor' - Penalty on df/du (default: 1e2)
            %   'initialConditionWeight' - Weight for initial condition (default: 1)
            
            obj.initialCondition = initialCondition;
            obj.odeFunction = odeFunction;
            obj.tspan = tspan;

            for i = 1:2:length(varargin)
                obj.(varargin{i}) = varargin{i + 1};
            end

            lossfunc = @(model, t) OneDim.IVPsolver.customLossFunction(model, t, initialCondition, odeFunction, 'penalty_factor', obj.penalty_factor, 'initialConditionWeight', obj.initialConditionWeight);

            model = OneDim.NN( ...
                'layerNum', 4, ...
                'neuronNumParOnelayer', 10, ...
                'domainStart', tspan(1), 'domainEnd', tspan(2), ...
                'initializationMethod','SIREN', ...
                'dataType', 'double'...
                );

            obj.optimizer = OneDim.Optimizer(model, 'auto', lossfunc, ...
                'numEpoch', 100, ...
                'numIterPerEpoch', 20, ...
                'batchSize', 128, ...
                'learnRate', 0.01, ...
                'convergenceThreshold', 1e-5, ...
                'earlyStopThreshold', 1e-6, ...
                'computeMode', 'cpu');
        end

        function obj = solve(obj)
            % SOLVE Train the neural network
            obj.optimizer = obj.optimizer.trainOneModel();
        end
    end

    methods (Static)
        function loss = customLossFunction(model, t, initialCondition, odeFunction, varargin)
            % CUSTOMLOSSFUNCTION Physics-informed loss function
            %
            % Computes: MSE(du/dt - f(u,t)) + C*(u(0) - u0)^2 + penalty
            
            [u, du] = model.d_ann(t);
            initValue = model.ann(model.domain(1));
            
            delta_u = 1e-10;
            dfdu = (odeFunction(u + delta_u, t) - odeFunction(u, t)) / delta_u;
            
            p = inputParser;
            addOptional(p, 'penalty_factor', 1e2);
            addOptional(p, 'initialConditionWeight', 1);
            parse(p, varargin{:});
            penalty_factor = p.Results.penalty_factor;
            initialConditionWeight = p.Results.initialConditionWeight;
            
            penalty = penalty_factor * max(0,mean(dfdu));
          
            loss = mean((du - odeFunction(u, t)).^2) + initialConditionWeight * (initValue - initialCondition)^2 + penalty;
        end
    end
end
