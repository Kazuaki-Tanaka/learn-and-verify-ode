classdef Optimizer
    % OPTIMIZER ADAM optimizer for training neural networks
    %
    % Supports single model and dual model training with region-based sampling.
    
    properties
        model
        numEpoch
        epochsCompleted = 0
        numIterPerEpoch
        batchSize
        learnRate
        iteration
        aveGrad
        aveSqGrad
        LossFunction
        trainingMode      % 'random', 'grid', 'region', or 'auto'
        convergenceThreshold
        earlyStopThreshold
        prevLoss
        computeMode       % 'cpu' or 'gpu'
        RandFunction
        patience = 5;
        patienceCounter = 0;
        regionCount = 10;
    end

    methods
        function obj = Optimizer(model, trainingMode, lossFunction, varargin)
            % OPTIMIZER Constructor
            %
            % Inputs:
            %   model - OneDim.NN or cell {NN, NN} for dual model training
            %   trainingMode - 'random', 'grid', 'region', or 'auto'
            %   lossFunction - Function handle for loss computation
            %
            % Optional Name-Value pairs:
            %   'numEpoch' - Number of epochs (default: 10)
            %   'numIterPerEpoch' - Iterations per epoch (default: 100)
            %   'batchSize' - Batch size (default: 32)
            %   'learnRate' - Learning rate (default: 0.001)
            %   'regionCount' - Number of regions for region mode (default: 10)
            
            if iscell(model) && numel(model) ~= 2
                error('Model cell array must contain exactly two elements.');
            end
            if ~isa(lossFunction, 'function_handle')
                error('LossFunction must be a function handle.');
            end

            % Default values
            defaultNumEpoch = 10;
            defaultNumIterPerEpoch = 100;
            defaultBatchSize = 32;
            defaultLearnRate = 0.001;
            defaultConvergenceThreshold = 1e-4;
            defaultEarlyStopThreshold = 1e-6;
            defaultComputeMode = 'cpu';
            defaultRandFunction = @rand;
            defaultPatience = 5;
            defaultRegionCount = 10;

            isValidFunction = @(f) isa(f, 'function_handle');

            p = inputParser;
            addRequired(p, 'model');
            addRequired(p, 'trainingMode', @(x) ismember(x,{'auto','random','grid','region'}));
            addParameter(p, 'numEpoch', defaultNumEpoch, @isnumeric);
            addParameter(p, 'numIterPerEpoch', defaultNumIterPerEpoch, @isnumeric);
            addParameter(p, 'batchSize', defaultBatchSize, @isnumeric);
            addParameter(p, 'learnRate', defaultLearnRate, @isnumeric);
            addParameter(p, 'convergenceThreshold', defaultConvergenceThreshold, @isnumeric);
            addParameter(p, 'earlyStopThreshold', defaultEarlyStopThreshold, @isnumeric);
            addParameter(p, 'computeMode', defaultComputeMode, @(x) ismember(x, {'cpu', 'gpu'}));
            addParameter(p, 'RandFunction', defaultRandFunction, isValidFunction);
            addParameter(p, 'patience', defaultPatience, @isnumeric);
            addParameter(p, 'regionCount', defaultRegionCount, @isnumeric);

            parse(p, model, trainingMode, varargin{:});

            obj.model = p.Results.model;
            obj.trainingMode = p.Results.trainingMode;
            obj.numEpoch = p.Results.numEpoch;
            obj.numIterPerEpoch = p.Results.numIterPerEpoch;
            obj.batchSize = p.Results.batchSize;
            obj.learnRate = p.Results.learnRate;
            obj.LossFunction = lossFunction;
            obj.convergenceThreshold = p.Results.convergenceThreshold;
            obj.earlyStopThreshold = p.Results.earlyStopThreshold;
            obj.computeMode = p.Results.computeMode;
            obj.RandFunction = p.Results.RandFunction;
            obj.patience = p.Results.patience;
            obj.regionCount = p.Results.regionCount;

            [obj.iteration, obj.aveGrad, obj.aveSqGrad, obj.prevLoss] = obj.initializeVariables();
        end

        function [iteration, aveGrad, aveSqGrad, prevLoss] = initializeVariables(obj)
            iteration = 0;
            if iscell(obj.model)
                aveGrad = cell(1, 2);
                aveSqGrad = cell(1, 2);
                prevLoss = {inf, inf};
            else
                aveGrad = [];
                aveSqGrad = [];
                prevLoss = inf;
            end
        end

        function relError = relativeError(~, oldVal, newVal)
            relError = abs(oldVal - newVal) / (abs(oldVal) + eps);
        end

        function relError = absoluteError(obj, oldVal, newVal)
            relError = abs(oldVal - newVal);
        end

        function obj = trainOneModel(obj)
            % TRAINONEMODEL Train a single neural network
            
            if strcmp(obj.computeMode, 'gpu')
                obj.model.parameters = cellfun(@(x) structfun(@gpuArray, x, 'UniformOutput', false), obj.model.parameters, 'UniformOutput', false);
            end

            % Generate training data based on mode
            if strcmp(obj.trainingMode, 'region')
                regionWidth = (obj.model.domain(2) - obj.model.domain(1)) / obj.regionCount;
                samplesPerRegion = ceil(obj.numIterPerEpoch * obj.batchSize / obj.regionCount);
                
                data = zeros(obj.regionCount, samplesPerRegion);
                for r = 1:obj.regionCount
                    regionStart = obj.model.domain(1) + (r - 1) * regionWidth;
                    regionEnd = obj.model.domain(1) + r * regionWidth;
                    data(r, :) = regionStart + (regionEnd - regionStart) * obj.RandFunction(1, samplesPerRegion);
                end
                data = reshape(data', numel(data), 1);
                data = data(1:obj.numIterPerEpoch * obj.batchSize);
                data = reshape(data, obj.numIterPerEpoch, obj.batchSize);
            elseif strcmp(obj.trainingMode, 'grid')
                grid = linspace(obj.model.domain(1), obj.model.domain(2), obj.numIterPerEpoch * obj.batchSize);
                data = reshape(grid, [obj.numIterPerEpoch, obj.batchSize]);
            elseif strcmp(obj.trainingMode, 'random') || strcmp(obj.trainingMode, 'auto')
                data = obj.model.domain(1) + (obj.model.domain(2) - obj.model.domain(1)) * obj.RandFunction(obj.numIterPerEpoch, obj.batchSize);
            else
                error('Invalid training mode: %s', obj.trainingMode);
            end

            % Add domain boundary points
            data = [repmat(obj.model.domain(1), obj.numIterPerEpoch, 1), data, repmat(obj.model.domain(2), obj.numIterPerEpoch, 1)];

            for epoch = 1:obj.numEpoch
                losses = zeros(1, obj.numIterPerEpoch);

                for iter = 1:obj.numIterPerEpoch
                    [gradVec, loss] = dlfeval(@obj.computeGradientsAndLoss, dlarray(data(iter,:)), obj.model.parameters);
                    losses(iter) = extractdata(loss);
                    obj = obj.updateOneModel(obj.model.parameters, gradVec);
                end

                epochLoss = mean(losses);

                % Early stopping
                if obj.absoluteError(obj.prevLoss, epochLoss) < obj.earlyStopThreshold
                    obj.patienceCounter = obj.patienceCounter + 1;
                    if obj.patienceCounter >= obj.patience
                        fprintf("Early stopping at epoch=%d\n", epoch);
                        break;
                    end
                else
                    obj.patienceCounter = 0;
                end

                % Auto mode: switch to grid when converged
                if strcmp(obj.trainingMode, 'auto') && obj.absoluteError(obj.prevLoss, epochLoss) < obj.convergenceThreshold
                    obj.trainingMode = 'grid';
                    grid = linspace(obj.model.domain(1), obj.model.domain(2), obj.numIterPerEpoch * obj.batchSize);
                    data = reshape(grid, [obj.numIterPerEpoch, obj.batchSize]);
                end

                obj.prevLoss = epochLoss;
                fprintf("epoch=%d, loss=%f, mode=%s\n", epoch, epochLoss, obj.trainingMode);
                pause(0.0001)
                obj.model.plot();
            end
            obj.epochsCompleted = epoch;

            if strcmp(obj.computeMode, 'gpu')
                % Properly gather parameters from GPU
                obj.model.parameters = cellfun(@(x) structfun(@gather, x, 'UniformOutput', false), ...
                    obj.model.parameters, 'UniformOutput', false);
            end
        end

        function obj = trainTwoModels(obj)
            % TRAINTWOMODELS Train two neural networks simultaneously
            
            if strcmp(obj.computeMode, 'gpu')
                obj.model{1}.parameters = cellfun(@(x) structfun(@gpuArray, x, 'UniformOutput', false), obj.model{1}.parameters, 'UniformOutput', false);
                obj.model{2}.parameters = cellfun(@(x) structfun(@gpuArray, x, 'UniformOutput', false), obj.model{2}.parameters, 'UniformOutput', false);
            end
        
            % Generate training data based on mode
            if strcmp(obj.trainingMode, 'region')
                regionWidth = (obj.model{1}.domain(2) - obj.model{1}.domain(1)) / obj.regionCount;
                samplesPerRegion = ceil(obj.numIterPerEpoch * obj.batchSize / obj.regionCount);

                if obj.regionCount > obj.numIterPerEpoch * obj.batchSize
                    error('regionCount cannot be greater than numIterPerEpoch * batchSize');
                end

                data = zeros(obj.regionCount, samplesPerRegion);
                for r = 1:obj.regionCount
                    regionStart = obj.model{1}.domain(1) + (r - 1) * regionWidth;
                    regionEnd = obj.model{1}.domain(1) + r * regionWidth;
                    data(r, :) = regionStart + (regionEnd - regionStart) * obj.RandFunction(1, samplesPerRegion);
                end

                data = reshape(data', numel(data), 1);
                data = data(1:obj.numIterPerEpoch * obj.batchSize);
                data = reshape(data, obj.numIterPerEpoch, obj.batchSize);
            elseif strcmp(obj.trainingMode, 'grid')
                grid = linspace(obj.model{1}.domain(1), obj.model{1}.domain(2), obj.numIterPerEpoch * obj.batchSize);
                data = reshape(grid, [obj.numIterPerEpoch, obj.batchSize]);
            elseif strcmp(obj.trainingMode, 'random') || strcmp(obj.trainingMode, 'auto')
                data = obj.model{1}.domain(1) + (obj.model{1}.domain(2) - obj.model{1}.domain(1)) * obj.RandFunction(obj.numIterPerEpoch, obj.batchSize);
            else
                error('Invalid training mode: %s', obj.trainingMode);
            end

            % Add domain boundary points
            data = [repmat(obj.model{1}.domain(1), obj.numIterPerEpoch, 1), data, repmat(obj.model{1}.domain(2), obj.numIterPerEpoch, 1)];

            if ~iscell(obj.aveGrad)
                obj.aveGrad = {obj.aveGrad, []};
            end
            if ~iscell(obj.aveSqGrad)
                obj.aveSqGrad = {obj.aveSqGrad, []};
            end
            if ~iscell(obj.prevLoss)
                obj.prevLoss = {obj.prevLoss, inf};
            end
        
            for epoch = 1:obj.numEpoch
                losses1 = zeros(1, obj.numIterPerEpoch);
                losses2 = zeros(1, obj.numIterPerEpoch);
        
                for iter = 1:obj.numIterPerEpoch
                    [gradVec1, gradVec2, loss1, loss2] = dlfeval(@obj.computeGradientsAndLoss2, dlarray(data(iter,:)), obj.model{1}.parameters, obj.model{2}.parameters);
                    losses1(iter) = extractdata(loss1);
                    losses2(iter) = extractdata(loss2);
                    obj = obj.updateTwoModels(obj.model{1}.parameters, obj.model{2}.parameters, gradVec1, gradVec2);
                end
        
                epochLoss1 = mean(losses1);
                epochLoss2 = mean(losses2);

                % Early stopping
                if obj.absoluteError(obj.prevLoss{1}, epochLoss1) < obj.earlyStopThreshold && ...
                   obj.absoluteError(obj.prevLoss{2}, epochLoss2) < obj.earlyStopThreshold
                    obj.patienceCounter = obj.patienceCounter + 1;
                    if obj.patienceCounter >= obj.patience
                        fprintf("Early stopping at epoch=%d\n", epoch);
                        break;
                    end
                else
                    obj.patienceCounter = 0;
                end

                % Auto mode: switch to grid when converged
                if strcmp(obj.trainingMode, 'auto') && ...
                    obj.absoluteError(obj.prevLoss{1}, epochLoss1) < obj.convergenceThreshold && ...
                    obj.absoluteError(obj.prevLoss{2}, epochLoss2) < obj.convergenceThreshold
                    obj.trainingMode = 'grid';
                    grid = linspace(obj.model{1}.domain(1), obj.model{1}.domain(2), obj.numIterPerEpoch * obj.batchSize);
                    data = reshape(grid, [obj.numIterPerEpoch, obj.batchSize]);
                end
                
                obj.prevLoss{1} = epochLoss1;
                obj.prevLoss{2} = epochLoss2;

                fprintf("epoch=%d, loss1=%f, loss2=%f, mode=%s\n", epoch, epochLoss1, epochLoss2, obj.trainingMode);
                pause(0.0001)
                
                subplot(1, 2, 1);
                obj.model{1}.plot();
                title('Model 1');

                subplot(1, 2, 2);
                obj.model{2}.plot();
                title('Model 2');              
            end
            obj.epochsCompleted = epoch;
        
            if strcmp(obj.computeMode, 'gpu')
                % Properly gather parameters from GPU
                obj.model{1}.parameters = cellfun(@(x) structfun(@gather, x, 'UniformOutput', false), ...
                    obj.model{1}.parameters, 'UniformOutput', false);
                obj.model{2}.parameters = cellfun(@(x) structfun(@gather, x, 'UniformOutput', false), ...
                    obj.model{2}.parameters, 'UniformOutput', false);
            end
        end

        function obj = updateOneModel(obj, parameters, gradVec)
            obj.iteration = obj.iteration + 1;
            [parameters, obj.aveGrad, obj.aveSqGrad] = adamupdate(parameters, gradVec, obj.aveGrad, obj.aveSqGrad, obj.iteration, obj.learnRate);
            obj.model.parameters = parameters;
        end

        function obj = updateTwoModels(obj, parameters1, parameters2, gradVec1, gradVec2)
            obj.iteration = obj.iteration + 1;

            [updatedParameters1, obj.aveGrad{1}, obj.aveSqGrad{1}] = adamupdate(parameters1, gradVec1, obj.aveGrad{1}, obj.aveSqGrad{1}, obj.iteration, obj.learnRate);
            obj.model{1}.parameters = updatedParameters1;

            [updatedParameters2, obj.aveGrad{2}, obj.aveSqGrad{2}] = adamupdate(parameters2, gradVec2, obj.aveGrad{2}, obj.aveSqGrad{2}, obj.iteration, obj.learnRate);
            obj.model{2}.parameters = updatedParameters2;
        end

        function [gradients, loss] = computeGradientsAndLoss(obj, x, params)
            if strcmp(obj.computeMode, 'gpu')
                x = gpuArray(x);
            end
            obj.model.parameters = params;
            loss = obj.LossFunction(obj.model, x);
            gradients = dlgradient(loss, obj.model.parameters);
        end

        function [gradients1, gradients2, loss1, loss2] = computeGradientsAndLoss2(obj, x, params1, params2)
            if strcmp(obj.computeMode, 'gpu')
                x = gpuArray(x);
            end

            obj.model{1}.parameters = params1;  
            obj.model{2}.parameters = params2;

            [loss1, loss2] = obj.LossFunction(obj.model{1}, obj.model{2}, x);

            gradients1 = dlgradient(loss1, params1);
            gradients2 = dlgradient(loss2, params2);
        end
    end
end
