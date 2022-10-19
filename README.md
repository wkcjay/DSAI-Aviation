[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OVqR_c_73r1QgubxiYTylDxg5Qmxy05Z)

# DSAI-Aviation
### Problem Statement
Predict the probability of death when taking a certain make and model of aircraft to certain location in which month and year
## Table of Contents
- [Dataset](#dataset)
  - [Download](#download)
- [Approaching Steps](#approaching-steps)
  - [Data Preparation](#data-preparation)
  - [Data Cleaning](#data-cleaning)
  - [Data Exploration and Analysis](#data-exploration-and-analysis)
  - [Machine Learning](#machine-learning)


### Dataset
#### Download
[Aviation Data](https://github.com/wkcjay/DSAI-Aviation/tree/main/DSAI-Aviation-Data)
### Approaching Steps
- Data Preparation
- Data Cleaning
- Data Exploration and Analysis
- Machine Learning

#### Data Preparation

```
#Choosing Data for the problem
MainData = pd.DataFrame(AviationData[['Injury.Severity','Location', 'Make', 'Model', 'Event.Date']])
MainData = MainData.dropna()
MainData = MainData.apply(lambda x: x.astype(str).str.upper())
```

#### Data Cleaning

```
#Clean 'Location'
Location = MainData['Location'].str.split(',').str[-1]
Location = Location.str.replace(" ", "")
MainData['Location'] = Location

Injury = DataExplore['Injury.Severity']

#Replace unnecessary letters to fit conventional names
Injury = Injury.str.replace('\d+', '')
Injury = Injury.str.replace(')', '')
Injury = Injury.str.replace('(', '')

#Convert to category
InjuryCat = Injury.astype('category')

# Replacing 'Injury.Severity' column with cleaned data
DataExplore['Injury.Severity'] = Injury

# Remove 'Unavailable' in 'Injury.Severity' column
New_Injury = DataExplore[(DataExplore['Injury.Severity'] != 'UNAVAILABLE') ]
New_Injury = New_Injury['Injury.Severity'].astype('category')
DataExplore['Injury.Severity'] = New_Injury
DataExplore = DataExplore.dropna()
```

#### Data Exploration and Analysis
- Top 10 Locations are in United States. California has the top most accident happened and the most Non-Fatal and Fatal. Florida has the most number of incident followed by Illinois.
<img src="https://github.com/wkcjay/DSAI-Aviation/blob/main/Figures/Location_vs_InjurySeverity.png" width="70%">

- Cessna has the most number of accident and the most number of Fatal and Non-Fatal Accident followed by Piper. Boeing has the most number of incident.
<img src="https://github.com/wkcjay/DSAI-Aviation/blob/main/Figures/Make_vs_InjurySeverity.png" width="70%">

- Since Cessna has the most number of accident report, majority of the top 10 aircraft model reported should be filled with Cessna aircraft model. However, one of the model from Piper has achieved the top 5.
<img src="https://github.com/wkcjay/DSAI-Aviation/blob/main/Figures/Make-Model_vs_InjurySeverity.png" width="70%">

- From the graph we can tell that throughout the years, the number of Fatal and Non Fatal Accidents are decreasing which clearly shows that the quality and safety of flights are improving. However, the incident rate are very linear throughout the years.
<img src="https://github.com/wkcjay/DSAI-Aviation/blob/main/Figures/Year_vs_InjurySeverity.png" width="70%">

- From the graph, we can tell that the top 5 is actually from May to September and July has the most amount of accident rate followed by August then June. Which also means there are higher chances of getting into accident during this period of time.
<img src="https://github.com/wkcjay/DSAI-Aviation/blob/main/Figures/Month_vs_InjurySeverity.png" width="70%">

#### Machine Learning
##### Label Encoding
```
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def LabelEncoding(var):
    x = array(Backup[var])
    # integer encode
    label_encoder = LabelEncoder()
    x_Encoded = label_encoder.fit_transform(x)
    Backup[var] = x_Encoded

List = ['Injury.Severity','Make','Model','Location']
for a in List:
    LabelEncoding(a)
```
##### RandomForestClassifier(random_state=42)
```
# Extract Response and Predictors
Class = pd.DataFrame(Backup["Injury.Severity"])

Labels = pd.DataFrame(Backup[["Make","Model","Month","Year","Location"]])

# Split the Dataset into Train and Test
Labels_train, Labels_test, Class_train, Class_test = train_test_split(Labels, Class, test_size = 0.25,random_state = 42)

rf = RandomForestClassifier(n_estimators=100,random_state = 42)
rf.fit(Labels_train,Class_train)
```
Classification Accuracy

Train : 0.9962126591866686

Test  : 0.7546276570562893

##### Predictor
```
def Predictor (a,b,c,d,e):
    
    MakerRow = MainData[MainData['Make'] == a].index[0]
    MakerData = Backup['Make'].index[MakerRow]
    ModelRow = MainData[MainData['Model'] == b].index[0]
    ModelData = Backup['Model'].index[ModelRow]
    LocationRow = MainData[MainData['Location'] == e].index[0]
    LocationData = Backup['Location'].index[LocationRow]
    Predicting = pd.DataFrame(np.array([[MakerData,ModelData,c,d,LocationData]]), columns=['Make', 'Model', 'Month', 'Year','Location'])
    Arr = rf.predict(Predicting)
    if Arr == [0]:
        Injury = 'FATAL'
    elif Arr == [1]:
        Injury = 'SAFE'
    else:
        Injury = 'SAFE'
    return Injury
```
