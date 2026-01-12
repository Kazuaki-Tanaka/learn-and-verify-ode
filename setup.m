function setup()
    % SETUP Add the package to the MATLAB path
    %
    % This script adds the package directory to the MATLAB search path.
    % It also checks for required dependencies.

    % Get the root directory of the package
    packagePath = fileparts(mfilename('fullpath'));
    
    % Add only the package root to path (packages starting with '+' are automatically handled)
    addpath(packagePath);
    
    fprintf('---------------------------------------------------\n');
    fprintf(' Learn and Verify ODE package\n');
    fprintf('---------------------------------------------------\n');
    fprintf('Package added to path: %s\n', packagePath);
    
    % Check for INTLAB
    if isempty(which('intvalinit'))
        fprintf('\n[Warning] INTLAB (Interval Laboratory) not found in path.\n');
        fprintf('This package requires INTLAB for rigorous verification.\n');
        fprintf('Download: http://www.ti3.tu-harburg.de/rump/intlab/\n');
    else
        fprintf('\n[OK] INTLAB found.\n');
    end

    % Check for Deep Learning Toolbox
    if isempty(which('dlarray'))
         fprintf('\n[Warning] Deep Learning Toolbox not found.\n');
         fprintf('This package requires Deep Learning Toolbox for neural network training.\n');
    else
         fprintf('[OK] Deep Learning Toolbox found.\n');
    end
    
    fprintf('---------------------------------------------------\n');
    
    % Ask to save path
    fprintf('\nTo use this package in future sessions without running setup,\n');
    fprintf('you can save the current path definition.\n');
    reply = input('Save path? Y/N [Y]: ', 's');
    
    if isempty(reply) || strcmpi(reply, 'Y')
        saveStatus = savepath;
        if saveStatus == 0
            fprintf('Path saved successfully.\n');
        else
            fprintf('Error saving path. You may need administrator privileges or check file permissions.\n');
        end
    else
        fprintf('Path not saved. You will need to run setup again in the next session.\n');
    end
end
