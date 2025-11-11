fID = fopen([filepath], 'w');

L_0_char = ['a'; 'b'; 'c'; 'd'; 'e'];
l0namebegin = ['\\newcommand{\\FF', steptag];
l0nameend1 = '}{%4.2f}\n';
l0nameend2 = '}{%4.0f}\n';

i = 1;
fprintf(fID, [l0namebegin 'EquityNoArb' l0nameend1], FF(i)); i = i + N + 1;
fprintf(fID, [l0namebegin 'BondNoArb' l0nameend1], FF(i)); i = i + N + 1;
fprintf(fID, [l0namebegin 'StockPD' l0nameend1], FF(i)); i = i + 1;
fprintf(fID, [l0namebegin 'GDPRP' l0nameend1], FF(i)); i = i + 1;
fprintf(fID, [l0namebegin 'DvdStrips' l0nameend1], sum(FF(i:i + 4))); i = i + 5;
fprintf(fID, [l0namebegin 'Bond' l0nameend1], sum(FF(i:i + 10))); i = i + 11;
fprintf(fID, [l0namebegin 'GoodDeal' l0nameend1], FF(i)); i = i + 1;
fprintf(fID, [l0namebegin 'Reg' l0nameend1], sum(FF(i:i + 39))); i = i + 40;

fprintf(fID, [l0namebegin 'Sum' l0nameend1], sum(FF)); i = i + 40;

fclose(fID);
