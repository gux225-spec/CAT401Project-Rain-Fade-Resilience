function RainFadeSimulator
%RAINFADESIMULATOR  Interactive ideal-physics rain fade & recovery simulator.
%   Base MATLAB only (R2020b or newer). Run:  >> RainFadeSimulator
%   Change rain rate, duration and link/network settings on the left; all plots update.
%   Tab 1: time response (rain, attenuation, capacity, users served for S1/S2/S3)
%   Tab 2: steady-state sweep over rain rate (fragility-style view)
%   Tab 3: model equations and assumptions
%   Files: rainFadeModel.m, ituP838.m, specEff.m, rainFadeDefaults.m, rainFadeSelfTest.m

D = rainFadeDefaults();
% NOAA Atlas 14 Vol.10 v3, NYC Central Park, PDS mean intensity (mm/h)
presets = {'Custom', NaN, NaN;
    '1-yr, 15-min (64 mm/h)',   64, 15;  '2-yr, 15-min (76 mm/h)',  76, 15;
    '10-yr, 15-min (112 mm/h)', 112, 15; '100-yr, 15-min (169 mm/h)', 169, 15;
    '1-yr, 60-min (27 mm/h)',   27, 60;  '10-yr, 60-min (48 mm/h)',  48, 60;
    '100-yr, 60-min (73 mm/h)', 73, 60;  '10-yr, 6-hr (15 mm/h)',    15, 360};
C = struct('S1', [0.45 0.45 0.45], 'S2', [0.90 0.50 0.05], 'S3', [0.13 0.60 0.20], ...
    'H', [0.00 0.45 0.74], 'F', [0.85 0.33 0.10], 'rain', [0.30 0.55 0.85]);

fig = uifigure('Name', 'Rain Fade & Recovery Simulator (ideal physics)', 'Position', [40 40 1480 880]);
main = uigridlayout(fig, [1 2]); main.ColumnWidth = {350, '1x'};

% ======================================================== controls
pnl = uipanel(main, 'Title', 'Scenario', 'FontWeight', 'bold');
g = uigridlayout(pnl, [32 2]); g.ColumnWidth = {'1.25x', '1x'};
g.RowHeight = repmat({23}, 1, 32); g.RowSpacing = 4; g.Scrollable = 'on';
row = 0; h = struct();

section('Rain event');
h.preset = dd('Preset (NOAA Atlas 14, NYC)', presets(:, 1)', 'Custom');
h.preset.ValueChangedFcn = @onPreset;
h.R   = spin('Peak rain rate (mm/h)', D.Rpeak, [0 250], 1);
h.Rs  = slide([0 250], D.Rpeak, h.R);
h.dur = spin('Rain duration (min)', D.durMin, [1 720], 1);
h.durs = slide([1 360], D.durMin, h.dur);
h.ramp = spin('Ramp up/down (min, 0 = step)', D.rampMin, [0 180], 1);
h.post = spin('Dry time after rain (min)', D.postMin, [5 600], 5);

section('Link physics');
h.pset = dd('Parameter set', {'Ideal physics', 'OpenMesh calibrated (Oct-Mar)', 'OpenMesh calibrated (Apr-Jun)'}, 'Ideal physics');
h.pset.ValueChangedFcn = @onParamSet;
calA = D.calA; wetR0 = D.wetR0;
h.fH  = spin('High band frequency (GHz)', D.fH, [1 100], 1);
h.fF  = spin('Fallback frequency (GHz)', D.fF, [0.6 100], 0.1);
h.L   = spin('Path length (km)', D.Lkm, [0.05 20], 0.1);
h.pol = dd('Polarisation', {'V', 'H'}, D.pol);
h.pr  = chk('ITU-R P.530 path reduction', D.usePathRed);
h.snrH = spin('Dry SNR high band (dB)', D.snr0H, [-5 50], 0.5);
h.snrF = spin('Dry SNR fallback (dB)', D.snr0F, [-5 50], 0.5);
h.wet = spin('Wet-antenna loss (dB, 0 = ideal)', D.wetMaxdB, [0 10], 0.5);
h.tdry = spin('Drying time constant (min)', D.tauDryMin, [0 120], 1);

section('Network & strategies');
h.hr  = spin('Headroom (dry cap / demand)', D.headroom, [0.5 5], 0.1);
h.S   = spin('Spare fallback capacity (%)', D.S * 100, [0 300], 5);
h.det = spin('Detection delay (s)', D.detectS, [0 600], 10);
h.hold = spin('S2 hold before switch-back (min)', D.holdMin, [0 120], 1);
h.tgt = spin('Service target (%)', D.target * 100, [50 100], 1);

row = row + 1;
b1 = uibutton(g, 'Text', 'Reset defaults', 'ButtonPushedFcn', @onReset); b1.Layout.Row = row; b1.Layout.Column = 1;
b2 = uibutton(g, 'Text', 'Export CSV + PNG', 'ButtonPushedFcn', @onExport); b2.Layout.Row = row; b2.Layout.Column = 2;
row = row + 1;
h.status = uilabel(g, 'Text', '', 'FontColor', [0.4 0.4 0.4], 'WordWrap', 'on');
h.status.Layout.Row = [row row + 1]; h.status.Layout.Column = [1 2];

% ======================================================== plots
tg = uitabgroup(main);
t1 = uitab(tg, 'Title', 'Time response');
t2 = uitab(tg, 'Title', 'Rain-rate sweep');
t3 = uitab(tg, 'Title', 'Model notes');

g1 = uigridlayout(t1, [5 1]); g1.RowHeight = {'0.75x', '1x', '1x', '1.4x', 112}; g1.RowSpacing = 2;
ax1 = uiaxes(g1); ax2 = uiaxes(g1); ax3 = uiaxes(g1); ax4 = uiaxes(g1);
tbl = uitable(g1, 'ColumnName', {'Strategy', 'Min service (%)', 'Minutes below target', ...
    'Lost user-minutes (EOM)', 'Recovery after rain stops (min)'}, 'RowName', {});

g2 = uigridlayout(t2, [2 1]);
axA = uiaxes(g2); axB = uiaxes(g2);

g3 = uigridlayout(t3, [1 1]);
uitextarea(g3, 'Editable', 'off', 'FontName', 'Consolas', 'Value', notesText());

update();

% ======================================================== callbacks / helpers
    function update(~, ~)
        P = readParams();
        o = rainFadeModel(P);
        x = o.tm;
        % --- rain
        cla(ax1); area(ax1, x, o.R, 'FaceColor', C.rain, 'EdgeColor', 'none', 'FaceAlpha', 0.7);
        ylabel(ax1, 'rain (mm/h)'); grid(ax1, 'on'); xlim(ax1, [x(1) x(end)]);
        title(ax1, sprintf('Peak %.0f mm/h, %.0f min, depth %.1f mm   |   rain stops at t = %.0f min', ...
            P.Rpeak, P.durMin, o.depthMm, o.rainEnd));
        % --- attenuation
        cla(ax2); hold(ax2, 'on');
        plot(ax2, x, o.AH, 'Color', C.H, 'LineWidth', 1.6);
        plot(ax2, x, o.AF, 'Color', C.F, 'LineWidth', 1.6);
        yline(ax2, o.Adeg, ':', sprintf('cannot carry demand (%.1f dB)', o.Adeg), 'Color', C.H, 'LabelHorizontalAlignment', 'left');
        yline(ax2, o.Aout, '--', sprintf('outage, SNR < %g dB (%.1f dB)', P.snrMin, o.Aout), 'Color', 'k', 'LabelHorizontalAlignment', 'left');
        hold(ax2, 'off'); grid(ax2, 'on'); xlim(ax2, [x(1) x(end)]);
        ylim(ax2, [0 max([o.Aout * 1.1, max(o.AH) * 1.1, 1])]);
        ylabel(ax2, 'extra path loss (dB)');
        legend(ax2, {sprintf('%.1f GHz (k=%.3g, \\alpha=%.3f)', P.fH, o.kH, o.aH), ...
            sprintf('%.1f GHz (k=%.3g, \\alpha=%.3f)', P.fF, o.kF, o.aF)}, 'Location', 'northeast');
        % --- capacity
        cla(ax3); hold(ax3, 'on');
        plot(ax3, x, o.capH, 'Color', C.H, 'LineWidth', 1.6);
        plot(ax3, x, o.capF, 'Color', C.F, 'LineWidth', 1.6);
        yline(ax3, 1, ':', 'basic demand', 'LabelHorizontalAlignment', 'left');
        hold(ax3, 'off'); grid(ax3, 'on'); xlim(ax3, [x(1) x(end)]);
        ylim(ax3, [0 max(P.headroom, P.S) * 1.15 + 0.05]);
        ylabel(ax3, 'capacity / demand');
        legend(ax3, {'high band capacity', 'fallback spare capacity'}, 'Location', 'northeast');
        % --- service
        cla(ax4); hold(ax4, 'on');
        plot(ax4, x, o.S1 * 100, 'Color', C.S1, 'LineWidth', 1.3);
        plot(ax4, x, o.S2 * 100, 'Color', C.S2, 'LineWidth', 1.3);
        plot(ax4, x, o.S3 * 100, 'Color', C.S3, 'LineWidth', 2.0);
        yline(ax4, P.target * 100, ':', 'target', 'LabelHorizontalAlignment', 'left');
        xline(ax4, o.rainStart, '-', 'rain starts', 'Color', [0.5 0.5 0.5]);
        xline(ax4, o.rainEnd, '-', 'rain stops', 'Color', [0.5 0.5 0.5]);
        hold(ax4, 'off'); grid(ax4, 'on'); xlim(ax4, [x(1) x(end)]); ylim(ax4, [-2 105]);
        xlabel(ax4, 'time (min)'); ylabel(ax4, 'users at min. service (%)');
        legend(ax4, o.names, 'Location', 'southeast');
        % --- table
        M = o.metrics; rc = cell(3, 5);
        for j = 1:3
            rc(j, :) = {o.names{j}, round(M(j, 1), 1), round(M(j, 2), 1), round(M(j, 3), 2), recTxt(M(j, 4))};
        end
        tbl.Data = rc;
        sweep(P);
        h.status.Text = sprintf('High band: demand lost above %.1f dB, outage above %.1f dB.  Total rain %.1f mm.', ...
            o.Adeg, o.Aout, o.depthMm);
    end

    function sweep(P)
        Rg = (0:0.5:250)';
        fl = unique([5.7 24 28 39 60 73 P.fH P.fF]);
        cla(axA); hold(axA, 'on');
        for f = fl
            [k, a] = ituP838(f, P.pol);
            lw = 1; if abs(f - P.fH) < 1e-9 || abs(f - P.fF) < 1e-9, lw = 2.2; end
            plot(axA, Rg, k * Rg.^a * P.Lkm, 'LineWidth', lw, 'DisplayName', sprintf('%.1f GHz', f));
        end
        o0 = rainFadeModel(setfield(P, 'wetMaxdB', 0)); %#ok<SFLD>
        yline(axA, o0.Adeg, ':', 'high band cannot carry demand', 'HandleVisibility', 'off');
        yline(axA, o0.Aout, '--', 'outage threshold', 'HandleVisibility', 'off');
        xline(axA, P.Rpeak, '-', 'current peak', 'HandleVisibility', 'off');
        hold(axA, 'off'); grid(axA, 'on'); ylim(axA, [0 max(o0.Aout * 1.5, 5)]);
        xlabel(axA, 'rain rate R (mm/h)'); ylabel(axA, sprintf('rain attenuation over %.2f km (dB)', P.Lkm));
        title(axA, 'ITU-R P.838-3 path attenuation (uniform rain)'); legend(axA, 'Location', 'northwest');

        [kH, aH] = ituP838(P.fH, P.pol); [kF, aF] = ituP838(P.fF, P.pol);
        gH = specEff(P.snr0H - kH * Rg.^aH * P.Lkm, P) / specEff(P.snr0H, P);
        gF = specEff(P.snr0F - kF * Rg.^aF * P.Lkm, P) / specEff(P.snr0F, P);
        capH = P.headroom * gH; capF = P.S * gF;
        s1 = min(1, capH);
        s2 = s1; s2(capH < 1) = min(1, capF(capH < 1));
        s3 = s1 + min(1 - s1, capF);
        cla(axB); hold(axB, 'on');
        plot(axB, Rg, s1 * 100, 'Color', C.S1, 'LineWidth', 1.3);
        plot(axB, Rg, s2 * 100, 'Color', C.S2, 'LineWidth', 1.3);
        plot(axB, Rg, s3 * 100, 'Color', C.S3, 'LineWidth', 2);
        iD = find(capH < 1, 1); iO = find(gH == 0, 1);
        if ~isempty(iD), xline(axB, Rg(iD), ':', sprintf('degraded > %.1f mm/h', Rg(iD)), 'HandleVisibility', 'off'); end
        if ~isempty(iO), xline(axB, Rg(iO), '--', sprintf('outage > %.1f mm/h', Rg(iO)), 'HandleVisibility', 'off'); end
        yline(axB, P.target * 100, ':', 'HandleVisibility', 'off');
        hold(axB, 'off'); grid(axB, 'on'); ylim(axB, [-2 105]);
        xlabel(axB, 'steady rain rate R (mm/h)'); ylabel(axB, 'users at min. service (%)');
        title(axB, sprintf('Steady state, %.1f GHz -> %.1f GHz, spare %.0f%%, headroom %.1fx', P.fH, P.fF, P.S * 100, P.headroom));
        legend(axB, {'S1 no redistribution', 'S2 link failover', 'S3 load-aware partial'}, 'Location', 'southwest');
    end

    function P = readParams()
        P = D;
        P.Rpeak = h.R.Value; P.durMin = h.dur.Value; P.rampMin = h.ramp.Value; P.postMin = h.post.Value;
        P.fH = h.fH.Value; P.fF = h.fF.Value; P.Lkm = h.L.Value; P.pol = h.pol.Value;
        P.usePathRed = h.pr.Value; P.snr0H = h.snrH.Value; P.snr0F = h.snrF.Value;
        P.wetMaxdB = h.wet.Value; P.tauDryMin = h.tdry.Value;
        P.calA = calA; P.wetR0 = wetR0;
        P.headroom = h.hr.Value; P.S = h.S.Value / 100; P.detectS = h.det.Value;
        P.holdMin = h.hold.Value; P.target = h.tgt.Value / 100;
        P.dt = max(5, min(60, round((P.preMin + P.durMin + P.postMin) * 60 / 3000)));
    end

    function onPreset(src, ~)
        i = find(strcmp(presets(:, 1), src.Value));
        if isnan(presets{i, 2}), return; end
        h.R.Value = presets{i, 2}; h.Rs.Value = min(250, presets{i, 2});
        h.dur.Value = presets{i, 3}; h.durs.Value = min(360, presets{i, 3});
        h.ramp.Value = 0;                       % Atlas 14 gives the mean intensity over the duration
        update();
    end

    function onParamSet(src, ~)
        % values from python/08_physics_calibration.py (OpenMesh NYC, 68-69 GHz + 5 GHz control)
        switch src.Value
            case 'Ideal physics'
                calA = 1; wetR0 = D.wetR0; h.wet.Value = 0; h.tdry.Value = D.tauDryMin;
            case 'OpenMesh calibrated (Oct-Mar)'
                calA = 1.06; wetR0 = 2.3; h.wet.Value = 1.35; h.tdry.Value = 29;
            otherwise
                calA = 0.68; wetR0 = 2.3; h.wet.Value = 1.35; h.tdry.Value = 29;
        end
        update();
    end

    function onReset(~, ~)
        h.pset.Value = 'Ideal physics'; calA = D.calA; wetR0 = D.wetR0;
        h.preset.Value = 'Custom';
        h.R.Value = D.Rpeak; h.Rs.Value = D.Rpeak; h.dur.Value = D.durMin; h.durs.Value = D.durMin;
        h.ramp.Value = D.rampMin; h.post.Value = D.postMin; h.fH.Value = D.fH; h.fF.Value = D.fF;
        h.L.Value = D.Lkm; h.pol.Value = D.pol; h.pr.Value = D.usePathRed; h.snrH.Value = D.snr0H;
        h.snrF.Value = D.snr0F; h.wet.Value = D.wetMaxdB; h.tdry.Value = D.tauDryMin;
        h.hr.Value = D.headroom; h.S.Value = D.S * 100; h.det.Value = D.detectS;
        h.hold.Value = D.holdMin; h.tgt.Value = D.target * 100;
        update();
    end

    function onExport(~, ~)
        P = readParams(); o = rainFadeModel(P);
        folder = uigetdir(pwd, 'Export folder');
        figure(fig);
        if isequal(folder, 0), return; end
        stamp = sprintf('R%g_D%g_f%g_S%g', P.Rpeak, P.durMin, P.fH, P.S * 100);
        T = table(o.tm, o.R, o.AH, o.AF, o.capH, o.capF, o.S1, o.S2, o.S3, 'VariableNames', ...
            {'time_min', 'rain_mmh', 'A_high_dB', 'A_fallback_dB', 'cap_high', 'cap_fallback', 'S1', 'S2', 'S3'});
        writetable(T, fullfile(folder, ['rainfade_' stamp '.csv']));
        try
            exportgraphics(t1, fullfile(folder, ['rainfade_' stamp '.png']), 'Resolution', 150);
        catch
            exportgraphics(ax4, fullfile(folder, ['rainfade_' stamp '_service.png']), 'Resolution', 150);
        end
        h.status.Text = ['Exported to ' folder];
    end

    % ---------- UI builders (rows are appended top to bottom)
    function section(txt)
        row = row + 1;
        l = uilabel(g, 'Text', txt, 'FontWeight', 'bold', 'FontColor', [0.1 0.3 0.6]);
        l.Layout.Row = row; l.Layout.Column = [1 2];
    end
    function c = spin(lbl, v, lim, step)
        row = row + 1;
        l = uilabel(g, 'Text', lbl); l.Layout.Row = row; l.Layout.Column = 1;
        c = uispinner(g, 'Value', v, 'Limits', lim, 'Step', step, 'ValueChangedFcn', @spinChanged);
        c.Layout.Row = row; c.Layout.Column = 2;
    end
    function s = slide(lim, v, linked)
        row = row + 1;
        s = uislider(g, 'Limits', lim, 'Value', min(max(v, lim(1)), lim(2)), 'MajorTicks', [], 'MinorTicks', []);
        s.Layout.Row = row; s.Layout.Column = [1 2];
        s.ValueChangedFcn = @(src, ~) sliderChanged(src, linked);
    end
    function c = dd(lbl, items, v)
        row = row + 1;
        l = uilabel(g, 'Text', lbl); l.Layout.Row = row; l.Layout.Column = 1;
        c = uidropdown(g, 'Items', items, 'Value', v, 'ValueChangedFcn', @update);
        c.Layout.Row = row; c.Layout.Column = 2;
    end
    function c = chk(lbl, v)
        row = row + 1;
        c = uicheckbox(g, 'Text', lbl, 'Value', v, 'ValueChangedFcn', @update);
        c.Layout.Row = row; c.Layout.Column = [1 2];
    end
    function spinChanged(src, ~)
        if src == h.R,   h.Rs.Value = min(h.Rs.Limits(2), src.Value);   h.preset.Value = 'Custom'; end
        if src == h.dur, h.durs.Value = min(h.durs.Limits(2), src.Value); h.preset.Value = 'Custom'; end
        update();
    end
    function sliderChanged(src, linked)
        linked.Value = round(src.Value);
        h.preset.Value = 'Custom';
        update();
    end
end

% ========================================================================
function s = recTxt(v)
if isnan(v), s = 'not within window'; else, s = sprintf('%.1f', v); end
end

function txt = notesText()
txt = {
'MODEL (ideal physics, deterministic)'
''
'1  Rain:        trapezoidal hyetograph R(t): ramp-up, plateau at peak, ramp-down.'
'                Uniform along the path unless the P.530 path-reduction box is ticked.'
'2  Attenuation: gamma = k * R^alpha  (dB/km), k, alpha from ITU-R P.838-3 (f, pol).'
'                A(t) = gamma * L * r  (+ optional wet-antenna loss Aw(t)).'
'                Aw relaxes to Wmax*(1-exp(-R/R0)) with tau_wet = 1 min while raining'
'                and tau_dry after the rain stops -> the only source of slow recovery'
'                in the physical layer (ideal default Wmax = 0: recovery is instant).'
'3  Radio:       SNR(t) = SNR0 - A(t);  SE = min(0.6*log2(1+SNR), 4.4) b/s/Hz,'
'                SE = 0 if SNR < -10 dB  (attenuated Shannon, 3GPP TR 36.942).'
'4  Capacity:    cap_H = headroom * SE_H(t)/SE_H0;  cap_F = S * SE_F(t)/SE_F0'
'                (both in units of the basic demand of the high-band users).'
'5  Strategies:  S1 no redistribution:   served = min(1, cap_H)'
'                S2 link failover:       all users -> fallback when measured cap_H < 1;'
'                                        back only after cap_H >= 1 for the hold time'
'                S3 load-aware partial:  keep min(1, measured cap_H) on H, overflow -> F'
'                "measured" = value delayed by the detection delay.'
'6  Metrics:     min service, minutes below target, lost user-minutes'
'                (EOM = integral of (1 - served) dt), recovery time after rain stops'
'                (until served >= 99 % for the rest of the window).'
''
'Presets: NOAA Atlas 14 Vol.10 v3 PDS mean intensities, NYC Central Park'
'(rectangular storm of that mean intensity and duration).'
''
'Limits: no gaseous/cloud attenuation change, no spatial rain structure (unless P.530),'
'no scheduler dynamics; headroom, spare capacity and SNR0 are user assumptions.'
'Defaults reproduce OpenMesh hop 22 (68 GHz, 1.8 km, SNR0 26 dB; 5.7 GHz, SNR0 19 dB).'
'Calibrated sets (step 8): A = a*ITU + W(R), W_max 1.35 dB, R0 2.3 mm/h, a = 1.06 (Oct-Mar)'
'or 0.68 (Apr-Jun, convective); drying tau = t50/ln2 = 29 min (median t50 = 20 min).'
'Validate with:  >> rainFadeSelfTest'
};
end
