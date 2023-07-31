# cicd-model-train-deploy-fastapi
A Continuous Integration / Continuous Delivery Model Release repo using FastAPI



# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Bryan Travis Smith created the mode.  It is a random forest model using
scikit-learn 1.3.0.   Parameters for the model are default values except for
those listed in params.yaml.

## Intended Use
The model is intended to to predice if the income of is larger atn 50K.  The
users are businesses assessing prospects for goods and services.

##  Data
The data was obtained from the UCI Machine Learning Repository
[link](https://archive.ics.uci.edu/dataset/20/census+income). The target class
as salary '>50K' as the positive target and all other values were negative
targets.

The data orignaly had 48842 rows and was split in a 80%/20% training/evaluation
set.

## Metrics
The model was evaluted using F1 score.  The value for the current mode is
0.6063

## Ethical Considerations

The model uses information related to protected classes: race, gender, and
martial status.

## Caveats and Recommendations

The model has very difference performance (f1 scores) on different categories.
Any use of the model should be coupled with validation of income and allow
disputes from the people being scored.

### Race
![race](https://raw.githubusercontent.com/bryantravissmith/cicd-model-train-deploy-fastapi/main/plots/race.png?raw=True)

### Sex
![sex](https://raw.githubusercontent.com/bryantravissmith/cicd-model-train-deploy-fastapi/main/plots/sex.png?raw=True)

### Relationship
![relationship](https://raw.githubusercontent.com/bryantravissmith/cicd-model-train-deploy-fastapi/main/plots/relationship.png?raw=True)