from enum import Enum
from fastapi import FastAPI
from joblib import load
import os
from pydantic import BaseModel


class WorkClass(str, Enum):
    StateGov = 'State-gov'
    SelfEmpNotInc = 'Self-emp-not-inc'
    Private = 'Private'
    FedGov = 'Federal-gov'
    LocalGov = 'Local-gov'
    Unknown = '?'
    SelfEmpInc = 'Self-emp-inc'
    WithoutPay = 'Without-pay'
    NeverWorked = 'Never-worked'


class Education(str, Enum):
    Bachelors = 'Bachelors'
    HSGrad = 'HS-grad'
    G11 = '11th'
    Masters = 'Masters'
    G9 = '9th'
    SomeCollege = 'Some-college'
    AssocAcdm = 'Assoc-acdm'
    AssocVoc = 'Assoc-voc'
    JrHigh = '7th-8th'
    PhD = 'Doctorate'
    ProfSchool = 'Prof-school'
    G5and6 = '5th-6th'
    G10 = '10th'
    G1thru4 = '1st-4th'
    Preschool = 'Preschool'
    G12 = '12th'


class MaritalStatus(str, Enum):
    NeverMarried = "Never-married"
    MarriedCivSpouse = "Married-civ-spouse"
    Divorced = "Divorced"
    MarriedSpouseAbsent = "Married-spouse-absent"
    Separated = "Separated"
    MarriedAFSpouse = "Married-AF-spouse"
    Widowed = "Widowed"


class Occupation(str, Enum):
    AdmClerical = "Adm-clerical"
    ExecManagerial = "Exec-managerial"
    HandlersCleaners = "Handlers-cleaners"
    ProfSpecialty = "Prof-specialty"
    OtherService = "Other-service"
    Sales = "Sales"
    CraftRepair = "Craft-repair"
    TransportMoving = "Transport-moving"
    FarmingFishing = "Farming-fishing"
    MachineOpInspct = "Machine-op-inspct"
    TechSupport = "Tech-support"
    Unknown = "?"
    ProtectiveServ = "Protective-serv"
    ArmedForces = "Armed-Forces"
    PrivHouseServ = "Priv-house-serv"


class Relationship(str, Enum):
    NotInFamily = "Not-in-family"
    Husband = "Husband"
    Wife = "Wife"
    OwnChild = "Own-child"
    Unmarried = "Unmarried"
    OtherRelative = "Other-relative"


class Race(str, Enum):
    White = "White"
    Black = "Black"
    AsianPacIslander = "Asian-Pac-Islander"
    AmerIndianEskimo = "Amer-Indian-Eskimo"
    Other = "Other"


class Sex(str, Enum):
    Male = "Male"
    Female = "Female"


class NativeCountry(str, Enum):
    UnitedStates = "United-States"
    Cuba = "Cuba"
    Jamaica = "Jamaica"
    India = "India"
    Unknown = "?"
    Mexico = "Mexico"
    South = "South"
    PuertoRico = "Puerto-Rico"
    Honduras = "Honduras"
    England = "England"
    Canada = "Canada"
    Germany = "Germany"
    Iran = "Iran"
    Philippines = "Philippines"
    Italy = "Italy"
    Poland = "Poland"
    Columbia = "Columbia"
    Cambodia = "Cambodia"
    Thailand = "Thailand"
    Ecuador = "Ecuador"
    Laos = "Laos"
    Taiwan = "Taiwan"
    Haiti = "Haiti"
    Portugal = "Portugal"
    DominicanRepublic = "Dominican-Republic"
    ElSalvador = "El-Salvador"
    France = "France"
    Guatemala = "Guatemala"
    China = "China"
    Japan = "Japan"
    Yugoslavia = "Yugoslavia"
    Peru = "Peru"
    OutlyingUS = "Outlying-US(Guam-USVI-etc)"
    Scotland = "Scotland"
    TrinadadAndTobago = "Trinadad&Tobago"
    Greece = "Greece"
    Nicaragua = "Nicaragua"
    Vietnam = "Vietnam"
    Hong = "Hong"
    Ireland = "Ireland"
    Hungary = "Hungary"
    HolandNetherlands = "Holand-Netherlands"


class Data(BaseModel):
    work_class: WorkClass
    education: Education


# Instantiate the app.
app = FastAPI()
model = load(os.path.join('model', 'random_forest_census_income.joblib'))


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}
