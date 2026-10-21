function se = specEff(snrdB, P)
%SPECEFF  Attenuated-Shannon spectral efficiency (3GPP TR 36.942), b/s/Hz.
se = min(P.seAlpha * log2(1 + 10.^(snrdB / 10)), P.seMax);
se(snrdB < P.snrMin) = 0;
end
