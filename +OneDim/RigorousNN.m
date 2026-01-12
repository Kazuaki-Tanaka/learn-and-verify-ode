classdef RigorousNN
    % RigorousNN - Rigorous verification of neural network solutions using interval arithmetic
    %
    % This class wraps an OneDim.NN model and provides interval arithmetic
    % operations using INTLAB for rigorous verification of ODE solutions.
    
    properties
        model;
        domain;
    end

    methods
        function obj = RigorousNN(model)
            if isa(model, 'OneDim.NN')
                obj.model = model;
                obj.domain = extractdata(model.domain);
            else
                error('Input model must be an instance of NN class.');
            end
        end

        function x = ann(obj, x)
            x = intval(x);
            t_input = x;
            x = obj.model.inputLayerActivation.f(t_input);
            n = size(obj.model.parameters, 2);
            W = cell(1, n);
            b = cell(1, n);
            for i = 1:n
                W{i} = extractdata(obj.model.parameters{i}.W);
                b{i} = extractdata(obj.model.parameters{i}.b);
            end
            for i = 1:n - 1
                x = W{i} * x + b{i};
                x = obj.model.activation{i}.f(x);
            end
            x = W{end} * x + b{end};
            x = obj.model.lastLayerActivation.f(x, t_input);
        end
        
        function [x,dx] = d_ann(obj, x)
            x = intval(x);
            t_input = x;
            dt_input = ones(size(x));
            
            x = obj.model.inputLayerActivation.f(t_input);
            dx = obj.model.inputLayerActivation.df(t_input);
            n = length(obj.model.parameters);
            W = cell(1, n);
            b = cell(1, n);
            for i = 1:n
                W{i} = extractdata(obj.model.parameters{i}.W);
                b{i} = extractdata(obj.model.parameters{i}.b);
            end
            for i = 1:n-1
                x = W{i} * x + b{i};
                dx = W{i} * dx;
                df = obj.model.activation{i}.df(x);
                x = obj.model.activation{i}.f(x);
                dx = df .* dx;
            end
            x = W{end} * x + b{end};
            dx = W{end} * dx;
 
            df_dx = obj.model.lastLayerActivation.df(x, t_input);
            df_dt = obj.model.lastLayerActivation.df_dt(x, t_input);
            dx = df_dx .* dx + df_dt .* dt_input;
            x = obj.model.lastLayerActivation.f(x, t_input);
        end
        
        function [x, dx, ddx] = dd_ann(obj, x)
            x = intval(x);
            t_input = x;
            dt_input = ones(size(x));
            ddt_input = zeros(size(x));
            
            x = obj.model.inputLayerActivation.f(t_input);
            dx = obj.model.inputLayerActivation.df(t_input);
            ddx = obj.model.inputLayerActivation.ddf(t_input);
            n = length(obj.model.parameters);
            W = cell(1, n);
            b = cell(1, n);
            for i = 1:n
                W{i} = extractdata(obj.model.parameters{i}.W);
                b{i} = extractdata(obj.model.parameters{i}.b);
            end
            for i = 1:n-1
                x = W{i} * x + b{i};
                ddx = W{i} * ddx;
                dx = W{i} * dx;
                df = obj.model.activation{i}.df(x);
                ddf = obj.model.activation{i}.ddf(x);
                d_old = dx;
                dx = df .* dx;
                ddx = ddf.*d_old.^2 + df.*ddx;
                x = obj.model.activation{i}.f(x);
            end
            x = W{end} * x + b{end};
            ddx = W{end} * ddx;
            dx = W{end} * dx;

            df_dx = obj.model.lastLayerActivation.df(x, t_input);
            df_dt = obj.model.lastLayerActivation.df_dt(x, t_input);
            ddf = obj.model.lastLayerActivation.ddf(x, t_input);
            ddf_dxdt = obj.model.lastLayerActivation.ddf_dxdt(x, t_input);
            ddf_dt2 = obj.model.lastLayerActivation.ddf_dt2(x, t_input);
            
            dx_old = dx;
            dx = df_dx .* dx + df_dt .* dt_input;
            ddx = ddf .* dx_old.^2 + 2 .* ddf_dxdt .* dx_old .* dt_input + df_dx .* ddx + ddf_dt2 .* dt_input.^2;
            x = obj.model.lastLayerActivation.f(x, t_input);
        end

        function [x, dx, ddx, dddx] = ddd_ann(obj, x)
            x = intval(x);
            t_input = x;
            dt_input = ones(size(x));
            
            x = obj.model.inputLayerActivation.f(t_input);
            dx = obj.model.inputLayerActivation.df(t_input);
            ddx = obj.model.inputLayerActivation.ddf(t_input);
            dddx = obj.model.inputLayerActivation.dddf(t_input);
            p = obj.model.parameters;
            act = obj.model.activation;
        
            n = length(p);
        
            W = cell(1, n);
            b = cell(1, n);
            for i = 1:n
                W{i} = extractdata(p{i}.W);
                b{i} = extractdata(p{i}.b);
            end
        
            for i = 1:n-1
                x = W{i} * x + b{i};
                dddx = W{i} * dddx;
                ddx = W{i} * ddx;
                dx = W{i} * dx;
        
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
            x = W{end} * x + b{end};
            dddx = W{end} * dddx;
            ddx = W{end} * ddx;
            dx = W{end} * dx;

            df_dx = obj.model.lastLayerActivation.df(x, t_input);
            df_dt = obj.model.lastLayerActivation.df_dt(x, t_input);
            ddf = obj.model.lastLayerActivation.ddf(x, t_input);
            ddf_dxdt = obj.model.lastLayerActivation.ddf_dxdt(x, t_input);
            ddf_dt2 = obj.model.lastLayerActivation.ddf_dt2(x, t_input);
            dddf_dx3 = obj.model.lastLayerActivation.dddf_dx3(x, t_input);
            dddf_dx2dt = obj.model.lastLayerActivation.dddf_dx2dt(x, t_input);
            dddf_dxdt2 = obj.model.lastLayerActivation.dddf_dxdt2(x, t_input);
            dddf_dt3 = obj.model.lastLayerActivation.dddf_dt3(x, t_input);
            
            dx_old = dx;
            ddx_old = ddx;
            dddx_old = dddx;
            
            dx = df_dx .* dx_old + df_dt .* dt_input;
            ddx = ddf .* dx_old.^2 + 2 .* ddf_dxdt .* dx_old .* dt_input + df_dx .* ddx_old + ddf_dt2 .* dt_input.^2;
            dddx = dddf_dx3 .* dx_old.^3 + 3 .* ddf .* dx_old .* ddx_old + 3 .* dddf_dx2dt .* dx_old.^2 ...
                 + 3 .* ddf_dxdt .* ddx_old .* dt_input + 3 .* dddf_dxdt2 .* dx_old .* dt_input.^2 ...
                 + df_dx .* dddx_old + dddf_dt3 .* dt_input.^3;
            
            x = obj.model.lastLayerActivation.f(x, t_input);
        end

        function [x, dx, ddx, dddx, ddddx] = dddd_ann(obj, x)
            x = intval(x);
            t_input = x;
            dt_input = ones(size(x));
            
            x = obj.model.inputLayerActivation.f(t_input);
            dx = obj.model.inputLayerActivation.df(t_input);
            ddx = obj.model.inputLayerActivation.ddf(t_input);
            dddx = obj.model.inputLayerActivation.dddf(t_input);
            ddddx = obj.model.inputLayerActivation.ddddf(t_input);
            p = obj.model.parameters;
            act = obj.model.activation;

            n = length(p);

            W = cell(1, n);
            b = cell(1, n);
            for i = 1:n
                W{i} = extractdata(p{i}.W);
                b{i} = extractdata(p{i}.b);
            end

            for i = 1:n-1
                x = W{i} * x + b{i};
                ddddx = W{i} * ddddx;
                dddx = W{i} * dddx;
                ddx = W{i} * ddx;
                dx = W{i} * dx;

                df = act{i}.df(x);
                ddf = act{i}.ddf(x);
                dddf = act{i}.dddf(x);
                ddddf = act{i}.ddddf(x);

                d_old = dx;
                dd_old = ddx;
                ddd_old = dddx;
                dx = df .* dx;
                ddx = ddf .* d_old.^2 + df .* ddx;
                dddx = dddf .* d_old.^3 + 3 * ddf .* d_old .* dd_old + df .* dddx;
                ddddx = ddddf.*d_old.^4 + 6*dddf.*d_old.^2.*dd_old + 3*ddf.*dd_old.^2 + 4*ddf.*d_old.*ddd_old + df.*ddddx;
                x = act{i}.f(x);
            end
            x = W{end} * x + b{end};
            ddddx = W{end} * ddddx;
            dddx = W{end} * dddx;
            ddx = W{end} * ddx;
            dx = W{end} * dx;

            df_dx = obj.model.lastLayerActivation.df(x, t_input);
            df_dt = obj.model.lastLayerActivation.df_dt(x, t_input);
            ddf = obj.model.lastLayerActivation.ddf(x, t_input);
            ddf_dxdt = obj.model.lastLayerActivation.ddf_dxdt(x, t_input);
            ddf_dt2 = obj.model.lastLayerActivation.ddf_dt2(x, t_input);
            dddf_dx3 = obj.model.lastLayerActivation.dddf_dx3(x, t_input);
            dddf_dx2dt = obj.model.lastLayerActivation.dddf_dx2dt(x, t_input);
            dddf_dxdt2 = obj.model.lastLayerActivation.dddf_dxdt2(x, t_input);
            dddf_dt3 = obj.model.lastLayerActivation.dddf_dt3(x, t_input);
            ddddf_dx4 = obj.model.lastLayerActivation.ddddf_dx4(x, t_input);
            ddddf_dx3dt = obj.model.lastLayerActivation.ddddf_dx3dt(x, t_input);
            ddddf_dx2dt2 = obj.model.lastLayerActivation.ddddf_dx2dt2(x, t_input);
            ddddf_dxdt3 = obj.model.lastLayerActivation.ddddf_dxdt3(x, t_input);
            ddddf_dt4 = obj.model.lastLayerActivation.ddddf_dt4(x, t_input);
            
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
            
            x = obj.model.lastLayerActivation.f(x, t_input);
        end

        function [ann_interval, d_ann_interval, dd_ann_interval, ddd_ann_interval] = preciseThirdDeriv(obj, x)
            % Taylor expansion for tighter interval bounds
            xm = mid(x);
            delta_x = x - xm;
            [ann_xm, d_ann_xm, dd_ann_xm, ddd_ann_xm] = obj.ddd_ann(xm);
            [~, ~, ~, dddd_ann_interval] = obj.dddd_ann(delta_x);
        
            ann_interval = ann_xm + d_ann_xm .* delta_x + 0.5 * dd_ann_xm .* (delta_x.^2) + (1/intval(6)) * ddd_ann_xm .* (delta_x.^3) + (1/intval(24)) * dddd_ann_interval .* (delta_x.^4);
            d_ann_interval = d_ann_xm + dd_ann_xm .* delta_x + 0.5 * ddd_ann_xm .* (delta_x.^2) + (1/intval(6)) * dddd_ann_interval .* (delta_x.^3);
            dd_ann_interval = dd_ann_xm + ddd_ann_xm .* delta_x + 0.5 * dddd_ann_interval .* (delta_x.^2);
            ddd_ann_interval = ddd_ann_xm + dddd_ann_interval .* delta_x;
        end

        function [ann_interval, d_ann_interval, dd_ann_interval] = preciseSecondDeriv(obj, x)
            xm = mid(x);
            delta_x = x - xm;
            [ann_xm, d_ann_xm, dd_ann_xm] = obj.dd_ann(xm);
            [~, ~, ~, ddd_ann_interval] = obj.ddd_ann(delta_x);
        
            ann_interval = ann_xm + d_ann_xm .* delta_x + 0.5 * dd_ann_xm .* (delta_x.^2) + (1/intval(6)) * ddd_ann_interval .* (delta_x.^3);
            d_ann_interval = d_ann_xm + dd_ann_xm .* delta_x + 0.5 * ddd_ann_interval .* (delta_x.^2);
            dd_ann_interval = dd_ann_xm + ddd_ann_interval .* delta_x;
        end
        
        function [ann_interval, d_ann_interval] = preciseFirstDeriv(obj, x)
            xm = mid(x);
            delta_x = x - xm;
            [ann_xm, d_ann_xm] = obj.d_ann(xm);
            [~, ~, dd_ann_interval] = obj.dd_ann(delta_x);
        
            ann_interval = ann_xm + d_ann_xm .* delta_x + 0.5 * dd_ann_interval .* (delta_x.^2);
            d_ann_interval = d_ann_xm + dd_ann_interval .* delta_x;
        end

        function [ann_interval] = preciseValue(obj, x)
            xm = mid(x);
            delta_x = x - xm;
        
            [ann_xm] = obj.ann(xm);
            [~, d_ann_interval] = obj.d_ann(delta_x);
        
            ann_interval = ann_xm + d_ann_interval .* delta_x;
        end
    end

    methods (Static)
        function result = odeVerifyer(vmodel, wmodel, umodel, odeFunction, initialCondition, n_initial)
            % ODEVERIFYER Rigorously verify sub- and super-solutions
            %
            % Verifies that u-v is a sub-solution and u+w is a super-solution.
            %
            % Inputs:
            %   vmodel - Neural network for sub-solution error (v >= 0)
            %   wmodel - Neural network for super-solution error (w >= 0)
            %   umodel - Neural network for approximate solution
            %   odeFunction - ODE function handle @(u, t)
            %   initialCondition - Initial value u(0)
            %   n_initial - Number of intervals for verification
            %
            % Output:
            %   result - Struct with fields:
            %     .verificationSuccess - true if verification succeeded
            %     .failReason - "none", "subsol violated", "supsol violated", or "max loop reached"
            %     .marginSubsol - Minimum margin for sub-solution condition
            %     .marginSupsol - Minimum margin for super-solution condition
            %     .wvUpperBound - Upper bound of (w+v) over the domain

            disp("Verifying sub-solution (u-v) and super-solution (u+w)...")
            vmodel = OneDim.RigorousNN(vmodel);
            wmodel = OneDim.RigorousNN(wmodel);
            umodel = OneDim.RigorousNN(umodel);

            result.verificationSuccess = false;
            result.failReason = "none";
            result.marginSubsol = NaN;
            result.marginSupsol = NaN;
            result.wvUpperBound = NaN;

            n = n_initial;
            x = linspace(vmodel.domain(1), vmodel.domain(2), n+1);
            X = infsup(x(1:end-1), x(2:end));
            X_max_diff = X;

            loop_count = 0;
            max_loop = 20;

            % Check initial conditions
            initival_value_u = umodel.preciseValue(umodel.domain(1));
            initival_value_v = vmodel.preciseValue(vmodel.domain(1));
            initival_value_w = wmodel.preciseValue(wmodel.domain(1));
            initival_value_subsol = initival_value_u - initival_value_v;
            initival_value_supsol = initival_value_u + initival_value_w;

            is_initial_condition_met_lower = (initialCondition >= initival_value_subsol.sup);
            if ~is_initial_condition_met_lower
                error('Initial condition not met: u(0) < (u-v)(0)');
            end

            is_initial_condition_met_upper = (initialCondition <= initival_value_supsol.inf);
            if ~is_initial_condition_met_upper
                error('Initial condition not met: u(0) > (u+w)(0)');
            end

            while true
                disp(['Iteration: ', num2str(loop_count), ', Intervals: ', num2str(length(X))]);

                [v, dv, ~, ~] = vmodel.preciseThirdDeriv(X);
                [w, dw, ~, ~] = wmodel.preciseThirdDeriv(X);
                [u, du, ~, ~] = umodel.preciseThirdDeriv(X);

                subsol = u - v;
                dsubsol = du - dv;
                supsol = u + w;
                dsupsol = du + dw;

                fsubsol = odeFunction(subsol, X);
                fsupsol = odeFunction(supsol, X);

                % Check sub-solution: dsubsol <= fsubsol
                if any(dsubsol.inf > fsubsol.sup)
                    disp('Sub-solution condition violated');
                    result.failReason = "subsol violated";
                    break;
                end

                % Check super-solution: dsupsol >= fsupsol
                if any(dsupsol.sup < fsupsol.inf)
                    disp('Super-solution condition violated');
                    result.failReason = "supsol violated";
                    break;
                end

                margin_subsol_all = fsubsol - dsubsol;
                margin_supsol_all = dsupsol - fsupsol;

                % Calculate maximum (w+v)
                [v_max_diff, ~, ~, ~] = vmodel.preciseThirdDeriv(X_max_diff);
                [w_max_diff, ~, ~, ~] = wmodel.preciseThirdDeriv(X_max_diff);
                diff = w_max_diff + v_max_diff;
                [max_diff_upper, max_ind] = max(diff.sup);
                max_ind_candidates = find(diff.sup >= diff(max_ind).inf);

                subsol_condition_idx = find(dsubsol.sup > fsubsol.inf);
                supsol_condition_idx = find(dsupsol.inf < fsupsol.sup);
                common_idx = unique([subsol_condition_idx, supsol_condition_idx]);

                if isempty(common_idx)
                    result.verificationSuccess = true;
                    result.failReason = "none";
                    result.marginSubsol = min(inf(margin_subsol_all));
                    result.marginSupsol = min(inf(margin_supsol_all));

                    disp(['Verified! (w+v) upper bound: ', num2str(max_diff_upper)]);
                    disp(['  Sub-solution margin: ', num2str(result.marginSubsol)]);
                    disp(['  Super-solution margin: ', num2str(result.marginSupsol)]);
                    break;
                end

                % Subdivide intervals that need refinement
                X_update = X(common_idx);
                X = [infsup(inf(X_update), mid(X_update)), infsup(mid(X_update), sup(X_update))];

                X_max_diff_update = X_max_diff(max_ind_candidates);
                X_max_diff = [infsup(inf(X_max_diff_update), mid(X_max_diff_update)), ...
                              infsup(mid(X_max_diff_update), sup(X_max_diff_update))];

                loop_count = loop_count + 1;
                if loop_count >= max_loop
                    disp('Maximum iterations reached');
                    result.failReason = "max loop reached";
                    break;
                end
            end

            if ~result.verificationSuccess
                if exist('margin_subsol_all','var') && exist('margin_supsol_all','var')
                    result.marginSubsol = min(inf(margin_subsol_all));
                    result.marginSupsol = min(inf(margin_supsol_all));
                end
            end

            if exist('max_diff_upper','var')
                result.wvUpperBound = max_diff_upper;
            end
        end
    end
end
