function P = rainFadeDefaults()
%RAINFADEDEFAULTS  Default scenario (hop 22 of OpenMesh: 69 GHz primary + 5.7 GHz fallback).

% --- rain event (ideal trapezoidal hyetograph)
P.Rpeak    = 40;     % peak rain rate (mm/h)
P.durMin   = 30;     % wet duration incl. ramps (min)
P.rampMin  = 5;      % ramp-up and ramp-down time (min); 0 = rectangular storm
P.preMin   = 20;     % dry time shown before rain (min)
P.postMin  = 60;     % dry time shown after rain (min)
P.dt       = 10;     % time step (s)

% --- link physics
P.fH       = 68;     % high (primary) band frequency (GHz)
P.fF       = 5.7;    % fallback band frequency (GHz)
P.Lkm      = 1.8;    % path length (km)
P.pol      = 'V';    % polarisation 'V' or 'H'
P.usePathRed = false;% ITU-R P.530 path reduction factor (false = uniform rain on path, ideal)
P.calA     = 1;      % multiplier on ITU rain attenuation (1 = ideal; OpenMesh Oct-Mar fit 1.06, Apr-Jun 0.68)
P.snr0H    = 26;     % dry SNR, high band (dB)  = RSL - noise floor
P.snr0F    = 19;     % dry SNR, fallback band (dB)
P.wetMaxdB = 0;      % wet-antenna loss at heavy rain (dB); 0 = ideal (no wet antenna)
P.wetR0    = 5;      % rain rate scale for wet-antenna saturation (mm/h)
P.tauWetMin = 1;     % wetting time constant (min)
P.tauDryMin = 10;    % drying time constant after rain stops (min)

% --- radio / capacity mapping (attenuated Shannon, 3GPP TR 36.942)
P.seAlpha  = 0.6;    % implementation efficiency
P.seMax    = 4.4;    % max spectral efficiency (b/s/Hz)
P.snrMin   = -10;    % below this SNR the link is in outage (dB)

% --- network / strategies
P.headroom = 1.5;    % dry high-band capacity / basic demand of its users
P.S        = 0.5;    % spare fallback capacity / basic demand of high-band users
P.detectS  = 60;     % detection + reassignment delay (s)
P.holdMin  = 10;     % S2: primary must stay healthy this long before switch-back (min)
P.target   = 0.95;   % service target (share of users at minimum service)
end
