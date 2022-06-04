from otree.api import *
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score, roc_auc_score, confusion_matrix, average_precision_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import shap
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.ensemble import RandomForestClassifier
import random
import statistics
import pickle


doc = """
One player decides how to divide a certain amount between himself and the other
player.
See: Kahneman, Daniel, Jack L. Knetsch, and Richard H. Thaler. "Fairness
and the assumptions of economics." Journal of business (1986):
S285-S300.
"""

# Initialize the base variables
class C(BaseConstants):
    NAME_IN_URL = 'dictator'
    PLAYERS_PER_GROUP = None     # No groups as we infer the data from each participant
    NUM_ROUNDS = 1
    INSTRUCTIONS_TEMPLATE = 'dictator_experiment_test/instructions.html'
    # Initial amount allocated to the dictator
    ENDOWMENT = cu(100)         # Endowment may be adapted


class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    treatment = models.IntegerField()                          # Is participant in treatment 0 or 1 ?
    dictator_prediction_full = models.IntegerField()           # Prediction based on all questionnaire attributes
    dictator_prediction_dec_fs = models.IntegerField()         # Prediction  based on selected questionnaire attributes
    kept = models.CurrencyField(                               # How much monetary value does the dictator keep?
        doc="""Amount dictator decided to keep for himself""",
        choices=[100, 70],
        label="I will keep",
    )

    feature_1 = models.IntegerField()   # Questionnaire formular field
    feature_2 = models.IntegerField()   # Questionnaire formular field
    feature_3 = models.IntegerField()   # Questionnaire formular field

    feature_1_dummy = models.IntegerField(choices=[[0, 'disclose'],[1, 'withhold']], label="Feature 1", widget=widgets.RadioSelect) # Dummy which adapts according to disclose or withhold decision
    feature_2_dummy = models.IntegerField(choices=[[0, 'disclose'],[1, 'withhold']], label="Feature 2", widget=widgets.RadioSelect) # Dummy which adapts according to disclose or withhold decision
    feature_3_dummy = models.IntegerField(choices=[[0, 'disclose'],[1, 'withhold']], label="Feature 3", widget=widgets.RadioSelect) # Dummy which adapts according to disclose or withhold decision

    belief_model_full   = models.IntegerField(choices=list(range(0,8)), label = "Prediction based on all attributes")
    belief_model_dec_fs = models.IntegerField(choices=list(range(0,8)), label = "Prediction based on selected attributes")

    BDM_full    = models.IntegerField(
        min=20, max=80, label="Please adjust the probability that the model overwrites your decision"
    )
    BDM_dec_fs = models.IntegerField(
        min=20, max=80, label="Please adjust the probability that the model overwrites your decision"
    )


# FUNCTIONS
def creating_session(subsession): # Assigns the experimental groups; itertools.cycle ensures that we have 50/50 distribution of treatment groups
    import itertools
    treatments = itertools.cycle([0, 1])
    for player in subsession.get_players():
        player.treatment=next(treatments)

def set_payoffs(player: Player):
    player.payoff = player.kept

testmodel = pickle.load(open('C:/Users/janmo/OneDrive/Dokumente/Goethe Uni/Doktor/Projekte/Decentralized Feature Selection 1/xgboost_otree_test_v2_impute.sav','rb'))

def predict_fairness_full(player: Player): # Predict the fairness/reciprocity of dictator based on all attributes

    # Initialize each dummy with 0; extent the number and names of dummies according to the used features
    player.feature_1_dummy = 0
    player.feature_2_dummy = 0
    player.feature_3_dummy = 0

    #Create the input for the ML model; consists of (1) questionnaire attr. and (2) dummies=0
    input_obs_dict = pd.DataFrame({"feature_1": player.feature_1,
                                   "feature_2": player.feature_2,
                                   "feature_3": player.feature_3,
                                   "feature_1_dummy": player.feature_1_dummy,
                                   "feature_2_dummy": player.feature_2_dummy,
                                   "feature_3_dummy": player.feature_3_dummy},
                                  index = [0])

    input_obs = pd.DataFrame(input_obs_dict) # Convert dict to DataFrame (only 1 row since we look at each single participant)
    player.dictator_prediction_full = int(testmodel.predict(input_obs)) # Perform the prediction


def predict_fairness_dec_fs(player: Player): # Predict the fairness/reciprocity of dictator after dec. FS

    list_medians = [18.0, 2319.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #  Todo: import list of medians automatically from other file
    # Next steps: automatisieren

    # Define the value of the features for the model; either the real feature value (disclose) or the training set
    # median (withhold)

    if player.feature_1_dummy == 1:
        model_input_1 = list_medians[0]
    else:
        model_input_1 = player.feature_1


    if player.feature_2_dummy == 1:
        model_input_2 = list_medians[1]
    else:
        model_input_2 = player.feature_2


    if player.feature_3_dummy == 1:
        model_input_3 = list_medians[2]
    else:
        model_input_3 = player.feature_3

    input_obs_dict = pd.DataFrame({"feature_1": model_input_1,
                                   "feature_2": model_input_2,
                                   "feature_3": model_input_3,
                                   "feature_1_dummy": player.feature_1_dummy,
                                   "feature_2_dummy": player.feature_2_dummy,
                                   "feature_3_dummy": player.feature_3_dummy},
                                  index=[0])

    input_obs = pd.DataFrame(input_obs_dict)
    player.dictator_prediction_dec_fs = int(testmodel.predict(input_obs))


# PAGES
class questionnaire(Page):
    form_model = 'player'
    form_fields = ['feature_1', 'feature_2', 'feature_3']

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        predict_fairness_full(player)

class Introduction(Page):
    pass

class dec_fs(Page):
    form_model = 'player'
    form_fields = ['feature_1_dummy', 'feature_2_dummy', 'feature_3_dummy'] #todo: Bei neuen features anpassen!

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        predict_fairness_dec_fs(player)

class Offer(Page):
    form_model = 'player'
    form_fields = ['kept']

class Results(Page):
    pass

class Introduction_of_algorithm(Page):
    pass

class Elicitation_of_model_beliefs(Page):
    form_model="player"
    form_fields= ["belief_model_full", "belief_model_dec_fs"]

class BDM(Page):
    form_model="player"
    form_fields= ["BDM_full", "BDM_dec_fs"]


page_sequence = [questionnaire,
                 Introduction,
                 Offer,
                 Introduction_of_algorithm,
                 dec_fs,
                 Elicitation_of_model_beliefs,
                 BDM,
                 Results]
