% calculate_forces.m - Calculates resulting forces from structure, support, and
% loading information
% Author: Leonardo White
% Team Members: Xingyang Liu, Ashley Morales, Leonardo White
% Date November 13 2026
% Inputs: connection matrix, X, Y, Sx, Sy, and L
% Outputs: force vector
% test/example structures
fig2_joint_cnt = 6;
fig2_member_cnt = 9;
fig2_C =   [1,0,1,0,0,0,0,0,0; % 9 members (columns), 6 joints (rows)
            1,1,0,1,0,0,0,0,0; % '1' if member m (column) is connected to joint j (row)
            0,1,0,0,1,1,0,0,0; % column sum should ALWAYS be 2
            0,0,1,1,1,0,1,1,0;
            0,0,0,0,0,1,1,0,1;
            0,0,0,0,0,0,0,1,1];
fig2_X =   [0,4,8,4,8,8]; % for (6) joints
fig2_Y =   [0,0,0,3,3,6];
hand_joint_cnt = 8;
hand_member_cnt = 13;
hand_C =   [1,1,0,0,0,0,0,0,0,0,0,0,0; %A %Columns:AB,AD,BC,BD,CD,CE,DE,DF,EF,EG,FG,FH,GH
            1,0,1,1,0,0,0,0,0,0,0,0,0; %B
            0,0,1,0,1,1,0,0,0,0,0,0,0; %C
            0,1,0,1,1,0,1,1,0,0,0,0,0; %D
            0,0,0,0,0,1,1,0,1,1,0,0,0; %E
            0,0,0,0,0,0,0,1,1,0,1,1,0; %F
            0,0,0,0,0,0,0,0,0,1,1,0,1; %G
            0,0,0,0,0,0,0,0,0,0,0,1,1];%H
hand_X =   [ 0, 0, 4, 4, 8, 8,12,12]; % meters
hand_Y =   [ 0, 4, 8, 4, 8, 4, 4, 0];
% test/example forces
fig2_Sx =  [1,0,0; % act upon (6) joints
            0,0,0; % script only supports exactly one entry == 1
            0,0,0;
            0,0,0;
            0,0,0;
            0,0,0];
fig2_Sy =  [0,1,0; % script only supports exactly two entries == 1
            0,0,0;
            0,0,1;
            0,0,0;
            0,0,0;
            0,0,0];
fig2_L =   [0;
            0;
            0;
            0;
            -120; % x load at joint 5
            -60;  % last x load
            0;    % y load start (joint 1)
            0;
            0;
            0;
            0;
            0];
hand_Sx =  [1,0,0;
            0,0,0;
            0,0,0;
            0,0,0;
            0,0,0;
            0,0,0;
            0,0,0;
            0,0,0];
hand_Sy =  [0,1,0;
            0,0,0;
            0,0,0;
            0,0,0;
            0,0,0;
            0,0,0;
            0,0,0;
            0,0,1];
hand_L =   [0;
            0;
            0;
            0;
            0;
            0;
            0;
            0;
            0; % y start
            0;
            0;
            -25; % Newtons
            0;
            0;
            0;
            0];
% loading inputs
joint_cnt = hand_joint_cnt; %fig2_joint_cnt;   % loading from testing/example data above
member_cnt = hand_member_cnt; %fig2_member_cnt; % ^ overwritten by any .mat's loaded below (here for testing)
unknown_cnt = 3; % script only supports exactly 2 unknown y_support and 1 unknown x_support forces
C = hand_C; %fig2_C;
X = hand_X; %fig2_X; % inches
Y = hand_Y; %fig2_Y;
Sx = hand_Sx; %fig2_Sx; % oz; exactly one entry = 1, all else = 0; 3 columns for 3 unknowns;   see ^
Sy = hand_Sy; %fig2_Sy; % oz; exactly two entries = 1, all else = 0; 3 columns for 3 unknowns; see ^
L = hand_L; %fig2_L;
% load in Liu's .mat file here, overwrites above
% preparing outputs
e = zeros(joint_cnt*2, member_cnt + unknown_cnt);
F = zeros(member_cnt + unknown_cnt);
member_lengths = zeros(member_cnt);
buckling_forces = zeros(member_cnt);
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
F = e\L; % = inv(e)*L\; internal member loads (+tension/-compression) followed by support forces
F;
for i = 1:1:(member_cnt)
    material_strength = 3391.3; % oz
    power_fit = -2.012; % from design manual (guessing it's from the class sample data)
    fit_error = 1.52; % oz
    buckling_forces(i) = material_strength * (member_lengths(i)^power_fit) - fit_error; % check
end
lowest_buckling_ratio_value = 1e37; % lowest --> closest to buckling
lowest_buckling_ratio_member = 0;
for i = 1:1:(member_cnt)
    if abs(buckling_forces(i)/F(i)) < lowest_buckling_ratio_value
        lowest_buckling_ratio_value = buckling_forces(i)/F(i);
        lowest_buckling_ratio_member = i;
    end
end
lowest_buckling_ratio_value;
lowest_buckling_ratio_member;
load = -sum(L,"all");
cost = sum(member_lengths,"all")*10;
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
%fprintf('Cost of truss: %.2f$ \n', max_load);
fprintf('Theoretical max load/cost ratio in oz/$: %.4f \n', max_load/cost); %TBD