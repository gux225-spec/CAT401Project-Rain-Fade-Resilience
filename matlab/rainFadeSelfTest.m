function rainFadeSelfTest()
%RAINFADESELFTEST  Checks the MATLAB model against ITU-R tables and the Python pipeline.
%   Expected scenario values were produced by an independent Python implementation of the
%   same equations (python/04_fallback_model.py uses the same ITU and SE functions).

[k, a] = ituP838(20, 'H'); assert(abs(k - 0.09164) < 1e-3 && abs(a - 1.0568) < 1e-3, 'P.838 20 GHz H');
[k, a] = ituP838(60, 'H'); assert(abs(k - 0.8606) < 2e-3 && abs(a - 0.7656) < 2e-3, 'P.838 60 GHz H');
[k, a] = ituP838(30, 'V'); assert(abs(k - 0.2291) < 1e-3 && abs(a - 0.9129) < 1e-3, 'P.838 30 GHz V');

out = rainFadeModel(rainFadeDefaults());
assert(abs(max(out.AH) - 26.048) < 0.01, 'peak 68 GHz attenuation');
assert(abs(max(out.AF) - 0.2310) < 0.001, 'peak 5.7 GHz attenuation');
expMin = [20.29 49.40 69.63];          % min service (%)
expEOM = [19.045 18.625 7.407];        % lost user-minutes
expBel = [26.5 37.5 25.667];           % minutes below 95 %
M = out.metrics;
assert(all(abs(M(:, 1)' - expMin) < 0.05), 'min service');
assert(all(abs(M(:, 3)' - expEOM) < 0.01), 'EOM');
assert(all(abs(M(:, 2)' - expBel) < 0.01), 'minutes below target');
% boundary case: hold time 0 must not chatter during steady rain (review item 6)
P = rainFadeDefaults(); P.holdMin = 0; P.rampMin = 0;
o = rainFadeModel(P);
inRain = o.tm > o.rainStart + 2 & o.tm < o.rainEnd - 1;
assert(sum(abs(diff(o.S2(inRain))) > 1e-9) == 0, 'S2 chatters with holdMin = 0');
assert(abs(mean(o.S2(inRain)) - min(1, P.S * mean(o.gF(inRain)))) < 1e-6, 'S2 should stay on fallback');
fprintf('rainFadeSelfTest: all checks passed.\n');
end
