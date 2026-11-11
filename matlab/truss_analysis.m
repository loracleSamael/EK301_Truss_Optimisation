function [member_forces, reactions, info] = truss_analysis(C, Sx, Sy, X, Y, L)
% truss_analysis.m - Calculates resulting forces from structure, support, and loading information
% Ported from calculate_forces.m (Author: Leonardo White; Team Members: Xingyang Liu, Ashley Morales,
% Leonardo White) as a function so the Python pipeline can call it. The maths is unchanged; the
% differences from calculate_forces.m are marked "CHANGED:" below.
%
% Inputs (six matrices, shapes for J joints and M members):
%   C   connection matrix, J x M (rows = joints, columns = members, 1 = connected)
%   Sx  x reaction matrix, J x 3 (exactly one entry = 1: the pin's x reaction, column 1)
%   Sy  y reaction matrix, J x 3 (exactly two entries = 1: pin y reaction column 2, roller column 3)
%   X   joint x locations [in], 1 x J
%   Y   joint y locations [in], 1 x J
%   L   external loads, 2J x 1 (rows 1..J are x loads, rows J+1..2J are y loads; a downward load
%       is NEGATIVE) [oz]
% Outputs:
%   member_forces  M x 1 member forces. SIGN CONVENTION OF THIS SCRIPT: e*F = L with a downward load
%                  negative, so a POSITIVE force is COMPRESSION and a negative force is tension.
%   reactions      3 x 1: Sx1 (pin x), Sy1 (pin y), Sy2 (roller y), same sign convention
%   info           struct: member_lengths, buckling_forces (capacity - fit_error), governing (member
%                  with the lowest buckling ratio, 1-based), ratio, load, max_load, cost
% The fprintf report is printed as in calculate_forces.m.

% CHANGED: inputs come from the arguments instead of the hard-coded example data
joint_cnt = size(C, 1);
member_cnt = size(C, 2);
unknown_cnt = 3; % script only supports exactly 2 unknown y_support and 1 unknown x_support forces
% preparing outputs
e = zeros(joint_cnt*2, member_cnt + unknown_cnt);
F = zeros(member_cnt + unknown_cnt);
member_lengths = zeros(member_cnt, 1); % CHANGED: column vector (was zeros(member_cnt), a square matrix)
buckling_forces = zeros(member_cnt, 1); % CHANGED: column vector
% algorithms
for i = 1:1:joint_cnt % from 1 to (e.g., 6) joints
    for j = 1:1:unknown_cnt % from 1 to (3) unknown external forces
        e(i, member_cnt + j) = Sx(i,j); % direct inserts Sx into e
    end
end
for i = 1:1:(joint_cnt)
    for j = 1:1:(unknown_cnt)
        e(i + joint_cnt, member_cnt + j) = Sy(i,j); % direct inserts Sy into e
    end
end
e; % check whether external support columns are correctly loaded into unit vector matrix
for j = 1:1:(member_cnt) % columns within connection matrix (not including the 3 "S" columns)
    joint_1 = -1;
    joint_2 = -1; % sum of each column always equals 2 for 2 joints / member in C
    for i = 1:1:(joint_cnt) % rows within connection matrix
        if C(i,j) == 1
            if joint_1 == -1 % if first joint in column hasn't been written-to-yet/found-until-now
                joint_1 = i;
            else
                joint_2 = i;
            end
        end
    end
    joint_x_distance = X(joint_2) - X(joint_1);
    joint_y_distance = Y(joint_2) - Y(joint_1);
    joint_r_distance = (joint_x_distance^2 + joint_y_distance^2)^0.5;
    member_lengths(j) = joint_r_distance;
    e(joint_1,j) = joint_x_distance / joint_r_distance;
    e(joint_2,j) = -joint_x_distance / joint_r_distance;
    e(joint_1 + joint_cnt,j) = joint_y_distance / joint_r_distance; % doing above but for y_comp lower half of e
    e(joint_2 + joint_cnt,j) = -joint_y_distance / joint_r_distance;
end
e; % for testing purposes
F = e\L; % = inv(e)*L\; internal member loads (+compression/-tension) followed by support forces
F;
for i = 1:1:(member_cnt)
    material_strength = 3391.3; % oz
    power_fit = -2.012; % from design manual (guessing it's from the class sample data)
    fit_error = 1.52; % oz (kept: the lower-bound capacity, so max_load is the conservative value)
    buckling_forces(i) = material_strength * (member_lengths(i)^power_fit) - fit_error; % check
end
lowest_buckling_ratio_value = 1e37; % lowest --> closest to buckling
lowest_buckling_ratio_member = 0;
for i = 1:1:(member_cnt)
    % CHANGED: only compression members (positive F in this script) can buckle. The original also
    % looked at tension members, which can wrongly be reported as the limit.
    if F(i) > 0
        if abs(buckling_forces(i)/F(i)) < lowest_buckling_ratio_value
            lowest_buckling_ratio_value = buckling_forces(i)/F(i);
            lowest_buckling_ratio_member = i;
        end
    end
end
lowest_buckling_ratio_value;
lowest_buckling_ratio_member;
load = -sum(L,"all");
cost = joint_cnt*10 + sum(member_lengths,"all"); % CHANGED: $10 per joint + $1 per inch of member
max_load = load * lowest_buckling_ratio_value;
fprintf('EK301, Section A3, Group 28: Xingyang Liu, Ashley Morales, Leonardo White, 11/13/2026 \n');
fprintf('Load: %.2f oz \n', load); % assumes loading forces in L are straight down Y
fprintf('Member forces in oz \n')
for i = 1:1:(member_cnt)
    if F(i) >= 0 % internal force = 0 --> tension/compression (floating-point precision issue)
        fprintf('m%d : %.3f (C) \n', i, F(i));
    else
        fprintf('m%d : %.3f (T) \n', i, -F(i));
    end
end
fprintf('Reaction forces in oz \n')
fprintf('Sx1 : %.3f \n', F(member_cnt+1));
fprintf('Sy1 : %.3f \n', F(member_cnt+2));
fprintf('Sy2 : %.3f \n', F(member_cnt+3));
fprintf('Cost of truss: %.2f$ \n', cost);
fprintf('Theoretical max load: %.3f oz (limited by member m%d) \n', max_load, lowest_buckling_ratio_member); % CHANGED: added so the limit is visible
fprintf('Theoretical max load/cost ratio in oz/$: %.4f \n', max_load/cost); %TBD

% CHANGED: outputs for the Python pipeline
member_forces = F(1:member_cnt);
reactions = F(member_cnt+1:member_cnt+unknown_cnt);
info = struct('member_lengths', member_lengths, 'buckling_forces', buckling_forces, ...
              'governing', lowest_buckling_ratio_member, 'ratio', lowest_buckling_ratio_value, ...
              'load', load, 'max_load', max_load, 'cost', cost);
end
