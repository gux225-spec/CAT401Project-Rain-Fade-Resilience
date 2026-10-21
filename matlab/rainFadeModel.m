function out = rainFadeModel(P)
%RAINFADEMODEL  Ideal-physics rain fade, capacity loss and recovery of a dual-band link.
%   out = rainFadeModel(P) with P from rainFadeDefaults(). Pure computation, no graphics.
%
%   Chain:  R(t) -> gamma = k R^alpha (ITU-R P.838-3) -> A = gamma * L * r (+ wet antenna)
%           -> SNR = SNR0 - A -> SE = min(a*log2(1+SNR), SEmax), 0 if SNR < SNRmin
%           -> capacity/demand -> users served under strategies S1/S2/S3.

dt = P.dt;
t  = (0:dt:(P.preMin + P.durMin + P.postMin) * 60)';
tm = t / 60;
n  = numel(t);

% ---------------- rain profile (trapezoid)
t0 = P.preMin; t1 = P.preMin + P.durMin;
ramp = min(P.rampMin, P.durMin / 2);
if ramp > 0
    R = P.Rpeak * min(max(min((tm - t0) / ramp, (t1 - tm) / ramp), 0), 1);
else
    R = P.Rpeak * double(tm >= t0 & tm < t1);
end

% ---------------- attenuation
[kH, aH] = ituP838(P.fH, P.pol);
[kF, aF] = ituP838(P.fF, P.pol);
if ~isfield(P, 'calA'), P.calA = 1; end
ArH = P.calA * rainAtten(R, kH, aH, P.Lkm, P.fH, P.usePathRed);
ArF = P.calA * rainAtten(R, kF, aF, P.Lkm, P.fF, P.usePathRed);
Aw  = wetAntenna(R, P);
AH  = ArH + Aw;
AF  = ArF + Aw;

% ---------------- capacity (relative to the basic demand of the high-band users)
gH = specEff(P.snr0H - AH, P) / specEff(P.snr0H, P);
gF = specEff(P.snr0F - AF, P) / specEff(P.snr0F, P);
capH = P.headroom * gH;
capF = P.S * gF;

% ---------------- strategies
nd   = min(round(P.detectS / dt), n - 1);
meas = [repmat(P.headroom, nd, 1); capH(1:end - nd)];   % what the controller sees

s1 = min(1, capH);                                        % S1 no redistribution

s2 = zeros(n, 1); onF = false; healthy = 0;               % S2 link-level failover
holdN = round(P.holdMin * 60 / dt);
for i = 1:n
    if ~onF && meas(i) < 1
        onF = true; healthy = 0;
    elseif onF
        if meas(i) >= 1, healthy = healthy + 1; else, healthy = 0; end
        if meas(i) >= 1 && healthy >= holdN, onF = false; end   % switch back only when primary is healthy
    end
    if onF, s2(i) = min(1, capF(i)); else, s2(i) = min(1, capH(i)); end
end

assign = min(1, meas);                                    % S3 load-aware partial
s3 = min(assign, capH) + min(1 - assign, capF);

% ---------------- metrics
names = {'S1 no redistribution'; 'S2 link failover'; 'S3 load-aware partial'};
SL = [s1 s2 s3];
M = zeros(3, 4);
for j = 1:3
    s = SL(:, j);
    bad = find(s < 0.99);
    if isempty(bad) || tm(bad(end)) < t1
        rec = 0;
    elseif bad(end) == n
        rec = NaN;                                        % not recovered in window
    else
        rec = tm(bad(end) + 1) - t1;
    end
    M(j, :) = [min(s) * 100, sum(s < P.target) * dt / 60, sum(1 - s) * dt / 60, rec];
end

% ---------------- thresholds (steady state, no wet antenna)
seT = specEff(P.snr0H, P) / P.headroom;                   % SE needed to carry all demand
if P.headroom <= 1
    Adeg = 0;
else
    Adeg = P.snr0H - 10 * log10(2^(seT / P.seAlpha) - 1);
end
Aout = P.snr0H - P.snrMin;

out = struct('t', t, 'tm', tm, 'R', R, 'AH', AH, 'AF', AF, 'Aw', Aw, 'gH', gH, 'gF', gF, ...
    'capH', capH, 'capF', capF, 'S1', s1, 'S2', s2, 'S3', s3, 'metrics', M, 'names', {names}, ...
    'kH', kH, 'aH', aH, 'kF', kF, 'aF', aF, 'Adeg', Adeg, 'Aout', Aout, ...
    'rainStart', t0, 'rainEnd', t1, 'depthMm', sum(R) * dt / 3600);
end

% =====================================================================
function A = rainAtten(R, k, a, d, f, usePathRed)
gamma = k * R.^a;
if usePathRed
    % ITU-R P.530-17 eq. (32); strictly defined for R0.01, used here as an approximation
    den = 0.477 * d^0.633 * max(R, 1e-3).^(0.073 * a) * f^0.123 - 10.579 * (1 - exp(-0.024 * d));
    r = 1 ./ den;
    r(den <= 0) = 2.5;
    r = min(max(r, 0), 2.5);
else
    r = 1;
end
A = gamma .* d .* r;
end

function Aw = wetAntenna(R, P)
n = numel(R); Aw = zeros(n, 1);
if P.wetMaxdB <= 0, return; end
tgt = P.wetMaxdB * (1 - exp(-R / P.wetR0));
for i = 2:n
    if tgt(i) > Aw(i - 1), tau = P.tauWetMin; else, tau = P.tauDryMin; end
    tau = max(tau, 1e-6) * 60;
    Aw(i) = tgt(i) + (Aw(i - 1) - tgt(i)) * exp(-P.dt / tau);
end
end
