[y,xt] = read_sparse('../heart_scale');
model=lrtrain(y, xt)
[l,a]=lrpredict(y, xt, model);

