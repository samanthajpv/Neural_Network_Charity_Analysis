# Neural_Network_Charity_Analysis
A neural network analysis on charity funding data

## Project Overview
The purpose of this project was to create a binary classifier to predict the success of investing in charities funded by a certain organization. Using tensorflow, together with scikit-learn and pandas, a number of optimization modifications were made to see if the model can achieve an accuracy of 75%. The charity dataset contains the following information:
- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding 
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application 
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

### Resources
- Data Source: [charity_data.csv](https://github.com/samanthajpv/Neural_Network_Charity_Analysis/blob/76c39e62a78b956d2eb72286f880e299deb169ae/charity_data.csv)
- Language: Python 3.7.10
    - Libraries: sci-kit learn, tensorflow, pandas
- Software: Jupyter Notebook, Google Colab
- Notebooks:
    - [AlphabetSoupCharity.ipynb](https://github.com/samanthajpv/Neural_Network_Charity_Analysis/blob/76c39e62a78b956d2eb72286f880e299deb169ae/AlphabetSoupCharity.ipynb)
    - [AlphabetSoupCharity_Optimization.ipynb](https://github.com/samanthajpv/Neural_Network_Charity_Analysis/blob/76c39e62a78b956d2eb72286f880e299deb169ae/AlphabetSoupCharity_Optimization.ipynb)

## Results
 
### Data Preprocessing
- The target variable of the model is the "IS_SUCCESSFUL" column. This tells if the money / investment was used effectively by the charities.
- The variables considered as features for the model are the following columns: "AFFILIATION", "CLASSIFICATION", "USE_CASE", "ORGANIZATION", "STATUS", "INCOME_AMT", "SPECIAL_CONSIDERATIONS", and "ASK_AMT". Columns with string data type were transformed using ```OneHotEncoder``` and data was normalized using ```StandardScaler```.
- The identification columns, namely "EIN" and "NAME", were removed from the input data. These were not considered as relevant variables or non-beneficial to the model.

### Compiling, Training, and Evaluating the Model
- There were two layers in the model. The first layer has 80 neurons and the second has 30. These numbers were chosen to have neurons higher and lower than the input features which is 44. The activation function for the hidden layers was 'ReLU' since it is known to outperform other functions. 'Sigmoid' was used for the output layer since model is for a binary classification.
- The target model performance is 75% but the model only produced an accuracy of 73.05%.

### Optimization
Below were steps taken to try and increase model performance:
#### Trial 1: Add neurons
The first modification was increasing the number of neurons for both layers (refer to the Optimization notebook). The accuracy decreased to 72.77%.
#### Trial 2: Add hidden layer
From the initial model, a third layer was added to see if it will increase the accuracy. Comparing to the first trial, accuracy increased to 72.86% but is still lower than the initial model.
#### Trial 3: Change activation
The third trial was changing the activation of the initial model to LeakyReLU. The model still did not reach the target performance with only 72.75%.
#### Trial 4: Revisit Features
Another modification done was to remove variables deemed to not have that much contribution to the model. Columns "STATUS" and "SPECIAL_CONSIDERATIONS" were dropped due to the count of its unique values. Although, model accuracy is still in the same range as the other trials with 72.79%.

## Summary
The model has an accuracy of 72-73% and did not reach the target performance of 75% even with several modifications. Some epoch did reach an accuracy of 74% but the model appears to be overfitting based on the printed epoch accuracies. One recommendation is to add more data since neural networks will perform better with more information. Another recommendation could be revisiting the features and eliminating noisy variables by feature selection or binning.

## Reference
(1) Trilogy Education Services. (2021, October). *Module 19 Challenge*. https://courses.bootcampspot.com/courses/626/assignments/13358?module_item_id=213881