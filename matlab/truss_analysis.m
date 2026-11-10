% truss_analysis.m
% PROVIDED FILE - place the supplied MATLAB analysis file here (do not rewrite).
%
% Inputs (six matrices):
%   - connection matrix (rows = joints, columns = members, 1 = connected)
%   - x reaction force matrix
%   - y reaction force matrix
%   - joint x locations
%   - joint y locations
%   - external loads matrix
% Outputs:
%   - member force matrix
%   - reaction force matrix
% Role: ground-truth analysis used to confirm the Python solver and the final designs.
