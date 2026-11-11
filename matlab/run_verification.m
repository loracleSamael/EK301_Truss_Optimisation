function run_verification(list_file)
% run_verification.m - batch driver used by the Python pipeline (src/matlab_io.py).
%
% list_file is a text file with one input .mat path per line (each holds C, X, Y, Sx, Sy, L as
% written by matlab_io.export_inputs). For every input <name>_input.mat this calls truss_analysis and
% writes
%   <name>_output.mat  member_forces, reactions, member_lengths, buckling_forces, governing,
%                      ratio, load, max_load, cost, error_message ('' when it worked)
%   <name>_output.txt  everything truss_analysis printed
% A failure in one design is recorded in error_message and does not stop the others.

here = fileparts(mfilename('fullpath'));
addpath(here);
paths = strtrim(readlines(list_file));
paths = paths(strlength(paths) > 0);
for k = 1:numel(paths)
    in_path = char(paths(k));
    out_mat = strrep(in_path, '_input.mat', '_output.mat');
    out_txt = strrep(in_path, '_input.mat', '_output.txt');
    error_message = '';
    member_forces = []; reactions = []; member_lengths = []; buckling_forces = [];
    governing = 0; ratio = 0; applied_load = 0; max_load = 0; cost = 0;
    diary(out_txt);
    fprintf('=== %s\n', in_path);
    try
        data = load(in_path);
        [member_forces, reactions, info] = truss_analysis(data.C, data.Sx, data.Sy, data.X, data.Y, data.L);
        member_lengths = info.member_lengths;
        buckling_forces = info.buckling_forces;
        governing = info.governing;
        ratio = info.ratio;
        applied_load = info.load;
        max_load = info.max_load;
        cost = info.cost;
    catch err
        error_message = err.message;
        fprintf('ERROR: %s\n', error_message);
    end
    diary off;
    save(out_mat, 'member_forces', 'reactions', 'member_lengths', 'buckling_forces', ...
         'governing', 'ratio', 'applied_load', 'max_load', 'cost', 'error_message');
end
end
