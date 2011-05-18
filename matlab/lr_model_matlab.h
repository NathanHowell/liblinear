const char *model_to_matlab_structure(mxArray *plhs[], int num_of_feature, struct lr_model *model);
const char *matlab_matrix_to_model(struct lr_model *model, const mxArray *matlab_struct);
