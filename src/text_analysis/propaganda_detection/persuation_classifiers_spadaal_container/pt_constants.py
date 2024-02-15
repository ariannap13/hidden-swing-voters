""" constants of the Persuasion Technique project

labels, convertion between different taxonomy, abbreviations
"""

"""
HOWTO USE

# fine: fine grained (as per inception and Guidelines)
# coarse: coarse (as per inception and Guidelines)
# stlabel: shared task labels 

# all functions have a boolean argument include_pity

from pt_constants import get_pt_labels, get_pt_label_converters, get_pt_nickname_converter

# labels list

list_pt_coarse = get_pt_labels("coarse")
list_pt_fine = get_pt_labels("fine")
list_pt_stlabel = get_pt_labels("fine", use_st_format=True)

# taxonomy <-> shared task label convertion

map_fine_stlabel, map_stlabel_fine = get_pt_label_converters()

# abbreviations

map_stlabel_abbrev = get_pt_nickname_converter("fine", use_st_format=True)
map_coarse_abbrev = get_pt_nickname_converter("coarse")
map_fine_abbrev = get_pt_nickname_converter("fine")
"""

import copy

### Labels

## raw taxonomy of inception

#

import pickle as pkl
label_dict = pkl.load(open('label_dict-golden.pkl','rb'))

list_framings = ["Economic",
"Capacity and resources",
"Morality",
"Fairness and equality",
"Legality, constitutionality and jurisprudence",
"Policy prescription and evaluation", 
"Crime and punishment", 
"Security and defense",
"Health and safety",
"Quality of life",
"Cultural identity",
"Public opinion",
"Political",
"External regulation and reputation", 
"Other"]

#

list_techniques = [
"Attack on Reputation",
"Attack on Reputation: Name calling or labeling",
"Attack on Reputation: Guilt by Association (Reductio ad Hitlerum)",
"Attack on Reputation: Casting Doubt",
"Attack on Reputation: Appeal to Hypocrisy (To quoque)",
"Attack on Reputation: Questioning the Reputation (Smears/Poisoning the Well)",
"Justification",
"Justification: Flag Waiving",
"Justification: Appeal to Authority",
"Justification: Appeal to Popularity (Bandwagon)",
"Justification: Appeal to Values",
"Justification: Appeal to fear, prejudice",
"Justification: Appeal to Pity",
"Distraction",
"Distraction: Misrepresentation of Someone’s Position (Strawman)",
"Distraction: Introducing irrelevant Information (Red Herring)",
"Distraction: Switching topic (Whataboutism)",
"Simplification",
"Simplification: Causal Oversimplification",
"Simplification: False dilemma or No Choice (Black-and-white Fallacy, Dictatorship)",
"Simplification: Consequential Oversimplification (Slippery slope)",
"Call",
"Call: Slogans",
"Call: Conversation killer (Thought-terminating cliché)",
"Call: Appeal to Time (Kairos)",
"Manipulative Wording",
"Manipulative Wording: Loaded Language",
"Manipulative Wording: Obfuscation, Intentional vagueness, Confusion",
"Manipulative Wording: Exaggeration or Minimisation",
"Manipulative Wording: Repetition",
"Other (unspecified)"
]

#
label_dict = {
0: 'Appeal_to_Authority',
1: 'Appeal_to_Fear-Prejudice',
2: 'Appeal_to_Hypocrisy',
3: 'Appeal_to_Popularity',
4: 'Appeal_to_Time',
5: 'Appeal_to_Values',
6: 'Causal_Oversimplification',
7: 'Consequential_Oversimplification',
8: 'Conversation_Killer',
9: 'Doubt',
10: 'Exaggeration-Minimisation',
11: 'False_Dilemma-No_Choice',
12: 'Flag_Waving',
13: 'Guilt_by_Association',
14: 'Loaded_Language',
15: 'Name_Calling-Labeling',
16: 'Obfuscation-Vagueness-Confusion',
17: 'Questioning_the_Reputation',
18: 'Red_Herring',
19: 'Repetition',
20: 'Slogans',
21: 'Straw_Man',
22: 'Whataboutism'}

label_dict_coarse = {
0:  'Attack on Reputation',
1: 'Justification',
2: 'Distraction',
3: 'Simplification',
4: 'Call',
5: 'Manipulative Wording'}

map_fineid2coarseid = {
0:1, #'Appeal_to_Authority:Justification',
1:1, #'Appeal_to_Fear-Prejudice:Justification'
2:0, #'Appeal_to_Hypocrisy:Attack on Reputation'
3:1, #'Appeal_to_Popularity:Justification'
4:4, #'Appeal_to_Time:Call'
5:2, #'Appeal_to_Values:Justification'
6:3, #'Causal_Oversimplification:Simplification'
7:3, #'Consequential_Oversimplification:Simplification'
8:4, #'Conversation_Killer:Call'
9:0, #'Doubt:Attack on Reputation'
10:5, #'Exaggeration-Minimisation:Manipulative Wording'
11:3, #'False_Dilemma-No_Choice:Simplification'
12:1, #'Flag_Waving:Justification'
13:0, #'Guilt_by_Association:Attack on Reputation'
14:5, #'Loaded_Language:Manipulative Wording'
15:0, #'Name_Calling-Labeling:Attack on Reputation'
16:5, #'Obfuscation-Vagueness-Confusion:Manipulative Wording'
17:0, #'Questioning_the_Reputation:Attack on Reputation'
18:2, #'Red_Herring:Distraction'
19:5, #'Repetition:Manipulative Wording'
20:4, #'Slogans:Call'
21:2, #'Straw_Man:Distraction'
22:2} #'Whataboutism:Distraction'

#

map_fine2coarse = { x+": "+y:x for x,y in [z.split(": ") for z in list_techniques if "Other" not in z and ": " in z]}
map_fine2coarse["Other (unspecified)"] = "Other (unspecified)"
map_fine2coarse

#

map_coarse2fine = {x:[] for x in map_fine2coarse.values()}
for fine,coarse in map_fine2coarse.items():
    map_coarse2fine[coarse].append(fine)
map_coarse2fine

#

list_pt_fine = [x for x in list_techniques if ": " in x]

#if not include_pity:
#    list_pt_fine = [x for x in list_pt_fine if x != 'Justification: Appeal to Pity']

list_pt_fine

#

list_pt_coarse = [x for x in list_techniques if ": " not in x and "Other" not in x]
list_pt_coarse

### Labels convertion

map_frame_stframe = {x:x.replace(" ", "_").replace(",", "").replace("_const", "_Const") for x in list_framings}

map_stframe_frame = {y:x for x,y in map_frame_stframe.items()}

map_frame_stframe

#

map_stlabel_label = {
"Appeal_to_Authority" : "Justification: Appeal to Authority", 
"Appeal_to_Fear-Prejudice" : "Justification: Appeal to fear, prejudice", 
"Appeal_to_Hypocrisy" : "Attack on Reputation: Appeal to Hypocrisy (To quoque)", 
"Appeal_to_Popularity" : "Justification: Appeal to Popularity (Bandwagon)", 
"Causal_Oversimplification" : "Simplification: Causal Oversimplification", 
"Conversation_Killer" : "Call: Conversation killer (Thought-terminating cliché)", 
"Doubt" : "Attack on Reputation: Casting Doubt", 
"Exaggeration-Minimisation" : "Manipulative Wording: Exaggeration or Minimisation", 
"False_Dilemma-No_Choice" : "Simplification: False dilemma or No Choice (Black-and-white Fallacy, Dictatorship)", 
"Flag_Waving" : "Justification: Flag Waiving", 
"Guilt_by_Association" : "Attack on Reputation: Guilt by Association (Reductio ad Hitlerum)", 
"Loaded_Language" : "Manipulative Wording: Loaded Language", 
"Name_Calling-Labeling" : "Attack on Reputation: Name calling or labeling", 
"Obfuscation-Vagueness-Confusion" : "Manipulative Wording: Obfuscation, Intentional vagueness, Confusion", 
"Red_Herring" : "Distraction: Introducing irrelevant Information (Red Herring)", 
"Repetition" : "Manipulative Wording: Repetition", 
"Slogans" : "Call: Slogans", 
"Straw_Man" : "Distraction: Misrepresentation of Someone’s Position (Strawman)", 
"Whataboutism" : "Distraction: Switching topic (Whataboutism)",
"Questioning_the_Reputation" : 'Attack on Reputation: Questioning the Reputation (Smears/Poisoning the Well)',
"Appeal_to_Time" : 'Call: Appeal to Time (Kairos)',
"Appeal_to_Pity" : 'Justification: Appeal to Pity',
"Appeal_to_Values" : 'Justification: Appeal to Values',
"Consequential_Oversimplification" : 'Simplification: Consequential Oversimplification (Slippery slope)'
}

map_label_stlabel = {y:x for x,y in map_stlabel_label.items()}

### Shared Task Labels

list_frames_st = list(map_stframe_frame.keys())

list_techniques_st = list(map_stlabel_label.keys())


# Nicknames

map_coarse_coarsenick = {'Attack on Reputation': "AR",
'Justification': "J",
'Distraction': "D",
'Simplification': "S",
'Call': "C",
'Manipulative Wording': "MW"}

map_coarsenick_coarse = {y:x for x,y in map_coarse_coarsenick.items()}

#

map_stlabel_nickname = { x : map_coarse_coarsenick[y.split(":")[0]] + ":" + "".join([z for z in x if z.isupper()]) for x,y in sorted(map_stlabel_label.items(), key=lambda x: list_pt_fine.index(x[1]))}
map_stlabel_nickname["Causal_Oversimplification"] = "S:CaO"
map_stlabel_nickname["Consequential_Oversimplification"] = "S:CoO"
map_stlabel_nickname['Appeal_to_Fear-Prejudice'] = "J:AF"
map_stlabel_nickname['Appeal_to_Pity'] = "J:APi"
map_stlabel_nickname['Appeal_to_Popularity'] = "J:APo"

map_stlabel_nickname

map_nickname_stlabel = {y:x for x,y in map_stlabel_nickname.items()}

map_fine_finenick = { y : map_coarse_coarsenick[y.split(":")[0]] + ":" + "".join([z for z in x if z.isupper()]) for x,y in sorted(map_stlabel_label.items(), key=lambda x: list_pt_fine.index(x[1]))}

map_fine_finenick[map_stlabel_label["Causal_Oversimplification"]] = "S:CaO"
map_fine_finenick[map_stlabel_label["Consequential_Oversimplification"]] = "S:CoO"
map_fine_finenick[map_stlabel_label['Appeal_to_Fear-Prejudice']] = "J:AF"
map_fine_finenick[map_stlabel_label['Appeal_to_Pity']] = "J:APi"
map_fine_finenick[map_stlabel_label['Appeal_to_Popularity']] = "J:APo"

map_finenick_fine = {y:x for x,y in map_fine_finenick.items()}


#

### GETTERS

def get_pt_labels(granularity, use_st_format=False, include_pity=True):
    if use_st_format:
        if granularity != "fine":
            raise Exception("incompatible arguments")
            return
        list_techniques_st_new = list_techniques_st.copy()
        if not include_pity:
            del list_techniques_st_new['Appeal_to_Pity']
        return list_techniques_st_new
        
    if granularity == "fine":
        list_pt_fine_new = list_pt_fine.copy()
        if not include_pity:
            list_pt_fine_new = [x for x in list_pt_fine_new if x != 'Justification: Appeal to Pity']
        return list_pt_fine_new
    elif granularity == "coarse":
        return list_pt_coarse
    else:
        raise Exception("invalid granularity: "+str(granularity))

def get_pt_level_converters(include_pity=True):
    ret = copy.deepcopy(map_fine2coarse), copy.deepcopy(map_coarse2fine)
    if not include_pity:
        del ret[0]['Justification: Appeal to Pity']
        print("R", ret[1]["Justification"])
        print("M", map_coarse2fine["Justification"])
        ret[1]["Justification"].remove('Justification: Appeal to Pity')
    return ret
        
def get_pt_label_converters(include_pity=True):
    map_stlabel_label_new = copy.deepcopy(map_stlabel_label)
    map_label_stlabel_new = copy.deepcopy(map_label_stlabel)
    if not include_pity:
        del map_stlabel_label_new['Appeal_to_Pity']
        del map_label_stlabel_new['Justification: Appeal to Pity']
    return map_label_stlabel_new, map_stlabel_label_new

def get_pt_nickname_converters(granularity, use_st_format=False, include_pity=True):
    if use_st_format:
        return copy.deepcopy(map_stlabel_nickname), copy.deepcopy(map_nickname_stlabel)
    
    if granularity == "coarse":
        return copy.deepcopy(map_coarse_coarsenick), copy.deepcopy(map_coarsenick_coarse)
    
    if granularity == "fine":
        return copy.deepcopy(map_fine_finenick), copy.deepcopy(map_finenick_fine)