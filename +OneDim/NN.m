classdef NN
    % NN - Fully-connected neural network with automatic differentiation
    %
    % Supports up to 4th order derivatives using manual chain rule implementation.
    % Uses SIREN initialization by default for sinusoidal activation functions.
    
    properties
        layerNum = 6;
        neuronNumParOnelayer = 50;
        activation;
        parameters;
        domain = [0, 1];
        initializationMethod = 'He';
        dataType = 'double';
        inputLayerActivation;
        outputDim = 1;
    end
    
    properties (Access = private)
        lastLayerActivation_internal;
    end
    
    properties (Dependent)
        lastLayerActivation;
    end

    methods
        function value = get.lastLayerActivation(obj)
            value = obj.lastLayerActivation_internal;
        end
        
        function obj = set.lastLayerActivation(obj, value)
            obj.lastLayerActivation_internal = obj.convertLastLayerActivation(value);
        end

        function obj = NN(varargin)
            % NN Constructor
            %
            % Optional Name-Value pairs:
            %   'layerNum' - Total layers (hidden + output), default: 6
            %   'neuronNumParOnelayer' - Neurons per hidden layer, default: 50
            %   'domainStart', 'domainEnd' - Input domain, default: [0, 1]
            %   'initializationMethod' - 'He', 'Xavier', 'SIREN', etc.
            %   'activation' - Custom activation functions
            %   'lastLayerActivation' - Output layer activation (struct with f, df, ...)
            
            p = inputParser;
            addOptional(p, 'layerNum', obj.layerNum);
            addOptional(p, 'neuronNumParOnelayer', obj.neuronNumParOnelayer);
            addOptional(p, 'domainStart', obj.domain(1));
            addOptional(p, 'domainEnd', obj.domain(2));
            addOptional(p, 'activationFunctions', {});
            addOptional(p, 'initializationMethod', obj.initializationMethod);
            addOptional(p, 'dataType', obj.dataType);

            % Default last layer activation (identity)
            defaultLastActivation = struct(...
                'f', @(x, t) x, ...
                'df', @(x, t) ones(size(x)), ...
                'df_dt', @(x, t) zeros(size(x)), ...
                'ddf', @(x, t) zeros(size(x)), ...
                'ddf_dxdt', @(x, t) zeros(size(x)), ...
                'ddf_dt2', @(x, t) zeros(size(x)), ...
                'dddf_dx3', @(x, t) zeros(size(x)), ...
                'dddf_dx2dt', @(x, t) zeros(size(x)), ...
                'dddf_dxdt2', @(x, t) zeros(size(x)), ...
                'dddf_dt3', @(x, t) zeros(size(x)), ...
                'ddddf_dx4', @(x, t) zeros(size(x)), ...
                'ddddf_dx3dt', @(x, t) zeros(size(x)), ...
                'ddddf_dx2dt2', @(x, t) zeros(size(x)), ...
                'ddddf_dxdt3', @(x, t) zeros(size(x)), ...
                'ddddf_dt4', @(x, t) zeros(size(x)));

            % Default input layer activation (identity)
            defaultInputActivation = struct(...
                'f', @(x) x, ...
                'df', @(x) ones(size(x)), ...
                'ddf', @(x) zeros(size(x)), ...
                'dddf', @(x) zeros(size(x)), ...
                'ddddf', @(x) zeros(size(x)));

            addOptional(p, 'lastLayerActivation', defaultLastActivation);
            addOptional(p, 'inputLayerActivation', defaultInputActivation);
            addOptional(p, 'outputDim', obj.outputDim);
            
            parse(p, varargin{:});
            
            obj.layerNum = p.Results.layerNum;
            obj.neuronNumParOnelayer = p.Results.neuronNumParOnelayer;
            obj.domain = dlarray([p.Results.domainStart, p.Results.domainEnd]);
            obj.initializationMethod = p.Results.initializationMethod;
            obj.dataType = p.Results.dataType;
            obj.lastLayerActivation = p.Results.lastLayerActivation;
            obj.inputLayerActivation = p.Results.inputLayerActivation;
            obj.outputDim = p.Results.outputDim;

            % Default: sinusoidal activation (SIREN)
            for i = 1 : obj.layerNum - 1
                obj.activation{i}.f = @sin;
                obj.activation{i}.df = @cos;
                obj.activation{i}.ddf = @(x) -sin(x);
                obj.activation{i}.dddf = @(x) -cos(x);                
                obj.activation{i}.ddddf = @sin;
                obj.activation{i}.derivative = @(x, n) OneDim.NN.nth_derivative_sin(x, n);
            end

            if ~isempty(p.Results.activationFunctions)
                for i = 1 : obj.layerNum - 1
                    obj.activation{i} = p.Results.activationFunctions{i};
                end
            end

            obj.parameters = obj.initializeWeights();
        end
        
        function converted = convertLastLayerActivation(obj, lastAct)
            % Convert 1-argument activation to 2-argument format
            if ~isfield(lastAct, 'df_dt')
                if ~isfield(lastAct, 'ddf'), lastAct.ddf = @(x) zeros(size(x)); end
                if ~isfield(lastAct, 'dddf'), lastAct.dddf = @(x) zeros(size(x)); end
                if ~isfield(lastAct, 'ddddf'), lastAct.ddddf = @(x) zeros(size(x)); end
                
                converted = struct(...
                    'f', @(x, t) lastAct.f(x), ...
                    'df', @(x, t) lastAct.df(x), ...
                    'df_dt', @(x, t) zeros(size(x)), ...
                    'ddf', @(x, t) lastAct.ddf(x), ...
                    'ddf_dxdt', @(x, t) zeros(size(x)), ...
                    'ddf_dt2', @(x, t) zeros(size(x)), ...
                    'dddf_dx3', @(x, t) lastAct.dddf(x), ...
                    'dddf_dx2dt', @(x, t) zeros(size(x)), ...
                    'dddf_dxdt2', @(x, t) zeros(size(x)), ...
                    'dddf_dt3', @(x, t) zeros(size(x)), ...
                    'ddddf_dx4', @(x, t) lastAct.ddddf(x), ...
                    'ddddf_dx3dt', @(x, t) zeros(size(x)), ...
                    'ddddf_dx2dt2', @(x, t) zeros(size(x)), ...
                    'ddddf_dxdt3', @(x, t) zeros(size(x)), ...
                    'ddddf_dt4', @(x, t) zeros(size(x)));
            else
                if ~isfield(lastAct, 'dddf_dx3'), lastAct.dddf_dx3 = @(x, t) zeros(size(x)); end
                if ~isfield(lastAct, 'ddddf_dx4'), lastAct.ddddf_dx4 = @(x, t) zeros(size(x)); end
                if ~isfield(lastAct, 'ddddf_dx3dt'), lastAct.ddddf_dx3dt = @(x, t) zeros(size(x)); end
                converted = lastAct;
            end
        end

        function plot(obj, outputIndex)
            % Plot the network output over the domain
            x = linspace(obj.domain(1), obj.domain(2), 1000);
            u = dlfeval(@obj.ann, dlarray(x));
            
            if nargin < 2
                plot(x, u);
            else
                plot(x, u(outputIndex, :));
            end
        end

        function x = ann(obj, x)
            % ANN Evaluate the neural network
            t_input = x;
            x = obj.inputLayerActivation.f(t_input);
            for i = 1 : size(obj.parameters,2) - 1
                x = fullyconnect(x, obj.parameters{i}.W, obj.parameters{i}.b, 'Dataformat', 'SB');
                x = obj.activation{i}.f(x);
            end
            x = fullyconnect(x, obj.parameters{end}.W, obj.parameters{end}.b, 'Dataformat', 'SB');
            x = obj.lastLayerActivation.f(x, t_input);
        end

        function [x,dx] = d_ann(obj, x)
            % D_ANN Evaluate network and its first derivative
            t_input = x;
            dt_input = ones(size(x));
            
            x = obj.inputLayerActivation.f(t_input);
            dx = obj.inputLayerActivation.df(t_input);
                        
            p = obj.parameters;
            act = obj.activation;
            n = length(p);
            
            for i = 1:n-1
                J = p{i}.W;
                x = fullyconnect(x, p{i}.W, p{i}.b, 'Dataformat', 'SB');
                dx = J * dx;
                df = act{i}.df(x);
                x = act{i}.f(x);
                dx = df .* dx;
            end
            x = fullyconnect(x, p{end}.W, p{end}.b, 'Dataformat', 'SB');
            dx = p{end}.W * dx;
            
            % Chain rule for lastLayerActivation
            df_dx = obj.lastLayerActivation.df(x, t_input);
            df_dt = obj.lastLayerActivation.df_dt(x, t_input);
            dx = df_dx .* dx + df_dt .* dt_input;
            x = obj.lastLayerActivation.f(x, t_input);
        end

        function [x,dx,ddx] = dd_ann(obj, x)
            % DD_ANN Evaluate network and its first two derivatives
            t_input = x;
            dt_input = ones(size(x));
            ddt_input = zeros(size(x));
            
            x = obj.inputLayerActivation.f(t_input);
            dx = obj.inputLayerActivation.df(t_input);
            ddx = obj.inputLayerActivation.ddf(t_input);

            p = obj.parameters;
            act = obj.activation;
            n = length(p);

            for i = 1:n-1
                J = p{i}.W;
                x = fullyconnect(x, p{i}.W, p{i}.b, 'Dataformat', 'SB');
                ddx = J * ddx;
                dx = J * dx;

                df = act{i}.df(x);
                ddf = act{i}.ddf(x);
                d_old = dx;
                dx = df .* dx;
                ddx = ddf.*d_old.^2 + df.*ddx;
                x = act{i}.f(x);
            end
            x = fullyconnect(x, p{end}.W, p{end}.b, 'Dataformat', 'SB');
            dx = p{end}.W * dx;
            ddx = p{end}.W * ddx;

            df_dx = obj.lastLayerActivation.df(x, t_input);
            df_dt = obj.lastLayerActivation.df_dt(x, t_input);
            ddf = obj.lastLayerActivation.ddf(x, t_input);
            ddf_dxdt = obj.lastLayerActivation.ddf_dxdt(x, t_input);
            ddf_dt2 = obj.lastLayerActivation.ddf_dt2(x, t_input);
            
            dx_old = dx;
            dx = df_dx .* dx + df_dt .* dt_input;
            ddx = ddf .* dx_old.^2 + 2 .* ddf_dxdt .* dx_old .* dt_input + df_dx .* ddx + ddf_dt2 .* dt_input.^2;
            x = obj.lastLayerActivation.f(x, t_input);
        end

        function [x, dx, ddx, dddx] = ddd_ann(obj, x)
            % DDD_ANN Evaluate network and its first three derivatives
            t_input = x;
            dt_input = ones(size(x));
            
            x = obj.inputLayerActivation.f(t_input);
            dx = obj.inputLayerActivation.df(t_input);            
            ddx = obj.inputLayerActivation.ddf(t_input);
            dddx = obj.inputLayerActivation.dddf(t_input);

            p = obj.parameters;
            act = obj.activation;
            n = length(p);

            for i = 1:n-1
                J = p{i}.W;
                x = fullyconnect(x, p{i}.W, p{i}.b, 'Dataformat', 'SB');
                dddx = J * dddx;
                ddx = J * ddx;
                dx = J * dx;

                df = act{i}.df(x);
                ddf = act{i}.ddf(x);
                dddf = act{i}.dddf(x);
                d_old = dx;
                dd_old = ddx;

                dx = df .* dx;
                ddx = ddf.*d_old.^2 + df.*ddx;
                dddx = dddf .* d_old.^3 + 3 * ddf.*d_old.*dd_old + df.*dddx; 
                x = act{i}.f(x);
            end
            x = fullyconnect(x, p{end}.W, p{end}.b, 'Dataformat', 'SB');
            dx = p{end}.W * dx;
            ddx = p{end}.W * ddx;
            dddx = p{end}.W * dddx;

            df_dx = obj.lastLayerActivation.df(x, t_input);
            df_dt = obj.lastLayerActivation.df_dt(x, t_input);
            ddf = obj.lastLayerActivation.ddf(x, t_input);
            ddf_dxdt = obj.lastLayerActivation.ddf_dxdt(x, t_input);
            ddf_dt2 = obj.lastLayerActivation.ddf_dt2(x, t_input);
            dddf_dx3 = obj.lastLayerActivation.dddf_dx3(x, t_input);
            dddf_dx2dt = obj.lastLayerActivation.dddf_dx2dt(x, t_input);
            dddf_dxdt2 = obj.lastLayerActivation.dddf_dxdt2(x, t_input);
            dddf_dt3 = obj.lastLayerActivation.dddf_dt3(x, t_input);
            
            dx_old = dx;
            ddx_old = ddx;
            dddx_old = dddx;
            
            dx = df_dx .* dx_old + df_dt .* dt_input;
            ddx = ddf .* dx_old.^2 + 2 .* ddf_dxdt .* dx_old .* dt_input + df_dx .* ddx_old + ddf_dt2 .* dt_input.^2;
            dddx = dddf_dx3 .* dx_old.^3 + 3 .* ddf .* dx_old .* ddx_old + 3 .* dddf_dx2dt .* dx_old.^2 ...
                 + 3 .* ddf_dxdt .* ddx_old .* dt_input + 3 .* dddf_dxdt2 .* dx_old .* dt_input.^2 ...
                 + df_dx .* dddx_old + dddf_dt3 .* dt_input.^3;
            
            x = obj.lastLayerActivation.f(x, t_input);
        end

        function [x, dx, ddx, dddx, ddddx] = dddd_ann(obj, x)
            % DDDD_ANN Evaluate network and its first four derivatives
            t_input = x;
            dt_input = ones(size(x));
            
            x = obj.inputLayerActivation.f(t_input);
            dx = obj.inputLayerActivation.df(t_input);            
            ddx = obj.inputLayerActivation.ddf(t_input);
            dddx = obj.inputLayerActivation.dddf(t_input);
            ddddx = obj.inputLayerActivation.ddddf(t_input);

            p = obj.parameters;
            act = obj.activation;
            n = length(p);

            for i = 1:n-1
                J = p{i}.W;
                x = fullyconnect(x, p{i}.W, p{i}.b, 'Dataformat', 'SB');
                ddddx = J * ddddx;
                dddx = J * dddx;
                ddx = J * ddx;
                dx = J * dx;

                df = act{i}.df(x);
                ddf = act{i}.ddf(x);
                dddf = act{i}.dddf(x);
                ddddf = act{i}.ddddf(x);
                d_old = dx;
                dd_old = ddx;
                ddd_old = dddx;

                dx = df .* dx;
                ddx = ddf.*d_old.^2 + df.*ddx;
                dddx = dddf .* d_old.^3 + 3 * ddf.*d_old.*dd_old + df.*dddx; 
                ddddx = ddddf.*d_old.^4 + 6*dddf.*d_old.^2.*dd_old + 3*ddf.*dd_old.^2 + 4*ddf.*d_old.*ddd_old + df.*ddddx;
                x = act{i}.f(x);
            end
            x = fullyconnect(x, p{end}.W, p{end}.b, 'Dataformat', 'SB');
            dx = p{end}.W * dx;
            ddx = p{end}.W * ddx;
            dddx = p{end}.W * dddx;
            ddddx = p{end}.W * ddddx;

            df_dx = obj.lastLayerActivation.df(x, t_input);
            df_dt = obj.lastLayerActivation.df_dt(x, t_input);
            ddf = obj.lastLayerActivation.ddf(x, t_input);
            ddf_dxdt = obj.lastLayerActivation.ddf_dxdt(x, t_input);
            ddf_dt2 = obj.lastLayerActivation.ddf_dt2(x, t_input);
            dddf_dx3 = obj.lastLayerActivation.dddf_dx3(x, t_input);
            dddf_dx2dt = obj.lastLayerActivation.dddf_dx2dt(x, t_input);
            dddf_dxdt2 = obj.lastLayerActivation.dddf_dxdt2(x, t_input);
            dddf_dt3 = obj.lastLayerActivation.dddf_dt3(x, t_input);
            ddddf_dx4 = obj.lastLayerActivation.ddddf_dx4(x, t_input);
            ddddf_dx3dt = obj.lastLayerActivation.ddddf_dx3dt(x, t_input);
            ddddf_dx2dt2 = obj.lastLayerActivation.ddddf_dx2dt2(x, t_input);
            ddddf_dxdt3 = obj.lastLayerActivation.ddddf_dxdt3(x, t_input);
            ddddf_dt4 = obj.lastLayerActivation.ddddf_dt4(x, t_input);
            
            dx_old = dx;
            ddx_old = ddx;
            dddx_old = dddx;
            ddddx_old = ddddx;
            
            dx = df_dx .* dx_old + df_dt .* dt_input;
            ddx = ddf .* dx_old.^2 + 2 .* ddf_dxdt .* dx_old .* dt_input + df_dx .* ddx_old + ddf_dt2 .* dt_input.^2;
            dddx = dddf_dx3 .* dx_old.^3 + 3 .* ddf .* dx_old .* ddx_old + 3 .* dddf_dx2dt .* dx_old.^2 ...
                 + 3 .* ddf_dxdt .* ddx_old .* dt_input + 3 .* dddf_dxdt2 .* dx_old .* dt_input.^2 ...
                 + df_dx .* dddx_old + dddf_dt3 .* dt_input.^3;
            
            % 4th derivative (Faa di Bruno formula)
            term1  =       ddddf_dx4       .* (dx_old.^4);
            term2  = 6  .* dddf_dx3        .* (dx_old.^2) .* ddx_old;
            term3  = 4  .* ddf             .*  dx_old .* dddx_old;
            term4  = 3  .* ddf             .* (ddx_old.^2);
            term5  =       df_dx           .*              ddddx_old;
            term6  = 4  .* ddddf_dx3dt     .* (dx_old.^3);
            term7  = 12 .* dddf_dx2dt      .*  dx_old .* ddx_old;
            term8  = 6  .* ddddf_dx2dt2    .* (dx_old.^2);
            term9  = 4  .* ddf_dxdt        .*              dddx_old;
            term10 = 6  .* dddf_dxdt2      .*              ddx_old;
            term11 = 4  .* ddddf_dxdt3     .*              dx_old;
            term12 =       ddddf_dt4;
            
            ddddx = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12;
            
            x = obj.lastLayerActivation.f(x, t_input);
        end

        function p = initializeWeights(obj)
            % Initialize network weights
            n = obj.neuronNumParOnelayer;
            p{1}.W = obj.initializeWeightsMethod([n 1]);
            p{1}.b = dlarray(zeros(n,1));
            for i=2:obj.layerNum-1
                p{i}.W = obj.initializeWeightsMethod([n n]);
                p{i}.b = dlarray(zeros(n,1));
            end
            p{obj.layerNum}.W = obj.initializeWeightsMethod([obj.outputDim n]);
            p{obj.layerNum}.b = dlarray(zeros(obj.outputDim,1));
        end
        
        function weights = initializeWeightsMethod(obj, sz)
            switch obj.initializationMethod
                case 'He'
                    weights = randn(sz) * sqrt(2/sz(2)) ./ (obj.domain(2)-obj.domain(1));
                case 'Xavier'
                    weights = randn(sz) * sqrt(2 / (sz(1) + sz(2)));
                case 'LeCun'
                    weights = randn(sz) * sqrt(1 / sz(2));
                case 'FFT'
                    randomSignal = randn(prod(sz), 1);
                    fftCoefficients = fft(randomSignal);
                    weights = reshape(real(fftCoefficients), sz);
                case 'SIREN'
                    weights = sqrt(6/sz(2)) * (2*rand(sz) - 1) ./ (obj.domain(2)-obj.domain(1));
                case 'Zero'
                    weights = zeros(sz);
                case 'AlmostZero'
                    weights = sqrt(6/sz(2)) * (2*rand(sz) - 1) ./ (obj.domain(2)-obj.domain(1));
                    if sz(1) == 1
                        weights = weights./1e-2;
                    end
                otherwise
                    error('Unknown initialization method.');
            end
            weights = cast(weights, obj.dataType);
            weights = dlarray(weights);
        end

        function [wCell,bCell] = getParameters(obj)
            % Get all weights and biases as cell arrays
            wCell = cell(1, obj.layerNum);
            bCell = cell(1, obj.layerNum);
            for i = 1:obj.layerNum
                wCell{i} = obj.parameters{i}.W(:);
                bCell{i} = obj.parameters{i}.b(:);
            end
        end
    end

    methods (Static)
        function result = nth_derivative_sin(x, n)
            % N-th derivative of sin(x)
            switch mod(n, 4)
                case 0
                    result = sin(x);
                case 1
                    result = cos(x);
                case 2
                    result = -sin(x);
                case 3
                    result = -cos(x);
            end
        end
    end
end
