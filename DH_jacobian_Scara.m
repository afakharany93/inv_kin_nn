% 3/2/2017 
% made by :  Ahmed Almetwaly 
% Get the homogenous transformation with DH Method for Scara Robot  
%----------------------------------------------------
% for Scara Robot (change parameters here for readability)
%
% i  thetai  di  ai  alphai
% 1  theta1  l1  l2     0 
% 2  theta_2  0  l3     180 
% 3  0        d3  0     0 
% 4  theta4   l4  0     0 
%
% theta_2 = theta2-(pi/2) 
% --------------------------------------------------------
% initializations 
clear ; clc ; close all ; 
%------------------------------------------------------- 

% General Form of Homogenous Transformation Matrix 
syms thetai  di  ai alphai  ;
Ti (thetai , di , ai, alphai)= ...
    [ cos(thetai) , -cos(alphai)*sin(thetai) , sin(alphai)*sin(thetai) , ai*cos(thetai) ; ...
      sin(thetai) , cos(alphai)*cos(thetai) , -sin(alphai)*cos(thetai) , ai*sin(thetai) ; ...
           0       ,       sin(alphai)      ,       cos(alphai)        ,       di       ; ...
           0       ,            0           ,            0             ,        1         ] ; 
%---------------------------------------------------------------------------------------------------

% Identify the Variables

% sub for our scara robot         
syms l1 l2 l3 l4 th1 th2 th3 th4 ; 
syms T1 T2 T3 T4 ; 
% note theta in rad
disp('Homogenous Transformation Matrix')

%--------------------------------------------------------------------------

% substitute here with DH parameters :) :)

T1 = Ti(th1,l1,0,pi/2) ;
T2 = Ti(th2,0,l2,0) ;
T3 = Ti(th3,0,l3,0) ;
T4 = Ti(th4,0,l4,0) ;

%--------------------------------------------------------------------------

% Forward Position Kinematics

% refer them to o 
disp('To o')
T1_0 = T1 ;
T2_0 = T1*T2 ;
T3_0 = T1*T2*T3 ;
T4_0 = T1*T2*T3*T4 ;

%--------------------------------------------------------------------------

% Forward Velocity Kinematics
% Jacoubian Matrix 

% jacobian = [cross(T1_0([1:3],3),(T4_0([1:3],4)-[0;0;0])) ,cross(T2_0([1:3],3),(T4_0([1:3],4)-T1_0([1:3],4))) , T3_0([1:3],3),cross(T4_0([1:3],3),(T4_0([1:3],4)-T3_0([1:3],4))) ; ... 
%             T1_0([1:3],3)                                       ,      T2_0([1:3],3)               ,                    [0 ;0; 0]     ,    T4_0([1:3],3)                            ] ;

        
% construct the rotation part here

Jw1 = T1_0(1:3,3);  % Rotary joint
Jw2 = T2_0(1:3,3);
Jw3 = T3_0(1:3,3);  
Jw4 = T4_0(1:3,3);

% construct the translation part here

Jv1 = cross([0; 0; 1] ,(T4_0(1:3,4) - [0;0;0]));      % Rotary joint (0)
Jv2 = cross(T1_0(1:3,3) ,(T4_0(1:3,4) - T1_0(1:3,4)));  
Jv3 = cross(T2_0(1:3,3) ,(T4_0(1:3,4) - T2_0(1:3,4)));  
Jv4 = cross(T3_0(1:3,3) ,(T4_0(1:3,4) - T3_0(1:3,4)));

% form the Jacobian matrix

Jacobian = [Jv1 ,Jv2 ,Jv3 ,Jv4 ;...
            Jw1 ,Jw2 ,Jw3 ,Jw4 ];

%--------------------------------------------------------------------------

% Inverse Velocity Kinematics
% Inverse Jacobian

%JINV = pinv(Jacobian);

%--------------------------------------------------------------------------

% Hints:
% use simplify(f(x)) to reduce complexity while reading
% use pinv(x) instead of inv(x) for non square matrix
% use simplify(x) if the values contained Conjuagate to correct them

T4_0 = simplify(T4_0);
Jacobian = simplify(Jacobian);
%JINV = simplify(JINV);