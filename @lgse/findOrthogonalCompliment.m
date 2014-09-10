function Q_new = findOrthogonalCompliment(Q_pca, q)
q = Q_pca * q;
q_compliment = orthcomp(q);
null_basis = null([q_compliment Q_pca]);
basisSize = size(q_compliment);
compliment_basis = q_compliment * null_basis(1:basisSize(2),:);
Q_new = orth(compliment_basis);
end




