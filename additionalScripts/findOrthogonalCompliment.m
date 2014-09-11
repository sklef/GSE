function [Q_new, S, q, flag] = findOrthogonalCompliment(Q_pca, q)
%q = q / norm(q);
q_compliment = orthcomp(Q_pca * q);
null_basis = null([q_compliment Q_pca]);
basisSize = size(q_compliment);
compliment_basis = q_compliment * null_basis(1:basisSize(2),:);
Q_new = orth(compliment_basis);
S = linsolve(Q_pca, [(Q_pca * q) Q_new]);
sizeValue = size(S);
if sizeValue(2) == 3
    q = [1, 0]';
    Q_new = Q_pca(:, 2);
    S = eye(2);
    flag = 1;
else
    flag = 0;
    
end
end





