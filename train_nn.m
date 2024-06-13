%%PREDICTOR CLASSES
predictor_data = load('predictor_array.mat')
og_predictor_array = predictor_data.predictor_array;

% Initialize an empty cell array to store the strings
predictor_array = cell(size(og_predictor_array, 1), size(og_predictor_array, 2));


% Iterate over each element in the 3D array
for i = 1:size(og_predictor_array, 1)
    for k = 1:size(og_predictor_array, 2)
        % Convert each element to a string and store it in the cell array
        predictor_array{i, k} = convertCharsToStrings(og_predictor_array(i, k, :));
    end
end


wavelengths_data = load('wavelengths_array.mat')

wavelengths_array = wavelengths_data.wavelengths_array;

predictor_variable_names = cell(size(wavelengths_array));

% Fill the cell array with the variable names as strings
for i = 1:length(wavelengths_array)
    predictor_variable_names{i} = sprintf('wavelength_%.6fnm', wavelengths_array(i));
end

for i = 1:length(predictor_variable_names)
    if i < 10
        disp(predictor_variable_names{i})
    end
end

% Convert NumPy array to table with specified variable names
predictor_array = string(predictor_array);
predictor_table = array2table(predictor_array, 'VariableNames', predictor_variable_names);

%%RESPONSE CLASSES
response_data = load('response_array.mat');
response_array = response_data.response_array;

func_groups_data = load('func_groups_array.mat')
func_groups_array = func_groups_data.func_groups_array;
func_groups_cell_array = cellstr(func_groups_array);

for i = 1:length(func_groups_cell_array)
    if isempty(func_groups_cell_array{i})
        func_groups_cell_array{i} = 'No functional groups';
    end
end

response_variable_names = cell(size(func_groups_cell_array));

for i = 1:length(func_groups_array)
    response_variable_names{i} = func_groups_cell_array{i};
end

% Convert NumPy array to table with specified variable names
response_table = array2table(response_array, 'VariableNames', response_variable_names);

% Define split ratios
train_ratio = 0.7;
test_ratio = 0.15;
val_ratio = 0.15;

%SPLIT PREDICTOR DATA INTO TRAINING, VALIDATION, AND TESTING PREDICTOR SETS
%Get the total number of rows
total_X_rows = height(predictor_table);

%Randomize the data
rng('default'); %ensures that the same random sequences will be produced each time this script is run
shuffled_predictor_table = predictor_table(randperm(total_X_rows), :);

%Calculate the number of rows for each set
X_train_rows = round(train_ratio*total_X_rows)
X_test_rows = round(test_ratio*total_X_rows)
X_val_rows = round(val_ratio*total_X_rows)

%Split the data
X_test = shuffled_predictor_table(1 : X_train_rows, :);
X_train = shuffled_predictor_table(X_train_rows + 1 : X_train_rows + X_test_rows, :);
X_val = shuffled_predictor_table(X_train_rows + X_test_rows + 1 : end, :);

%SPLIT RESPONSE DATA INTO TRAINING, VALIDATION, AND TESTING RESPONSE SETS
%Get the total number of rows
total_Y_rows = height(response_table);

%Randomize the data
rng('default'); %ensures that the same random sequences will be produced each time this script is run
shuffled_response_table = response_table(randperm(total_Y_rows), :);

%Calculate the number of rows for each set
Y_train_rows = round(train_ratio*total_Y_rows)
Y_test_rows = round(test_ratio*total_Y_rows)
Y_val_rows = round(val_ratio*total_Y_rows)

%Split the data
Y_test = shuffled_response_table(1 : Y_train_rows, :);
Y_train = shuffled_response_table(Y_train_rows + 1 : Y_train_rows + Y_test_rows, :);

Y_val = shuffled_response_table(Y_train_rows + Y_test_rows + 1 : end, :);


data_table = horzcat(predictor_table, response_table);
%data_train = horzcat(X_train, Y_train);
data_train = [X_train, Y_train];
data_test = horzcat(X_test, Y_test);
%data_val = horzcat(X_val, Y_val);

%TRAIN THE MODEL
% Initialize cell array to hold networks for each class
nets = cell(1, total_Y_rows) %the number of response classes

for k = 1:total_Y_rows
    %Extract the k-th column of the response table as the response variable
    response = Y_train{:, k}; %Extract all rows of the k-th column
    response = categorical(response);
    nets{k} = fitcnet(data_train, response, "Standardize", true)
end


predictions = zeros(1, total_X_rows); %the number of predictor classes

for k = 1: total_X_rows
    predictions(k) = predict(nets{k}, data_test)
end

disp(predictions)


