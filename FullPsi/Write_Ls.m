fID = fopen([filepath], 'w');

L_0_char = ['a'; 'b'; 'c'; 'd'; 'e'];
l0namebegin = ['\\newcommand{\\Psi', steptag, 'LamC'];
l0nameend1 = '}{%4.2f}\n';
l0nameend2 = '}{%4.0f}\n';

L0_val = [L0(1); L0(3:5); L0(7)];
% L0 handles
for i = 1:5
    fprintf(fID, [l0namebegin L_0_char(i) l0nameend1], L0_val(i));
end

% L1 handles
L_1_char = ['a'; 'b'; 'c'; 'd'; 'e'; 'f'; 'g'; 'h'; 'i'; 'j'; 'k'];

l1namebegin = ['\\newcommand{\\Psi', steptag, 'LamT'];
l1nameend1 = '}{%4.2f}\n';
l1nameend2 = '}{%4.0f}\n';

fprintf(fID, [l1namebegin, 'aa}{%4.2f}\n'], L1(1, 1));
fprintf(fID, [l1namebegin, 'ab}{%4.2f}\n'], L1(1, 2));
fprintf(fID, [l1namebegin, 'ac}{%4.2f}\n'], L1(1, 3));
fprintf(fID, [l1namebegin, 'ad}{%4.2f}\n'], L1(1, 4));

% L1 handles row 3
for i = 1:11
    fprintf(fID, [l1namebegin 'c' L_1_char(i) l1nameend1], L1(3, i));
end

% L1 handles row 4
for i = 1:11
    fprintf(fID, [l1namebegin 'd' L_1_char(i) l1nameend1], L1(4, i));
end

% L1 handles row 7
for i = 1:11
    fprintf(fID, [l1namebegin 'g' L_1_char(i) l1nameend1], L1(7, i));
end

fclose(fID);
