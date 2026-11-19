"""
The lexical resources the NLP pipeline runs on.

Three layers, kept separate on purpose:

1. **Canonical vocabulary** -- the 87 model features, loaded from the trained
   artifacts so the lexicon can never drift from what the classifier expects.
2. **Surface variants** -- `ml/data/symptom_vocab.json` (clinical synonyms)
   plus colloquial English phrasings a patient actually types.
3. **Code-mixed variants** -- romanized Hindi ("bukhar", "pet dard") and
   Devanagari, because the target users of an Indian health app routinely
   type Hinglish. This is the layer that makes the app usable beyond
   textbook English, and it is handled as text normalization rather than
   translation, so no network call is involved.
"""

import json
import os
from functools import lru_cache

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml", "data")
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml", "model")


# ---------------------------------------------------------------------------
# Colloquial English phrasings, keyed by canonical model feature.
# ---------------------------------------------------------------------------

COLLOQUIAL_ALIASES: dict[str, list[str]] = {
    "joint_pain": ["knee pain", "knees hurt", "joints hurt", "knee hurts",
                   "pain in joints", "achy joints", "aching joints", "sore joints"],
    "joint_swelling": ["swollen knees", "knees swollen", "joints swollen",
                       "swollen joints", "swelling in joints", "knee is swollen",
                       "knees are swollen", "joints are swollen",
                       "swelling in my knees", "swollen and stiff"],
    "joint_stiffness": ["stiff joints", "stiff in the morning", "knee stiffness",
                        "morning stiffness", "joints feel stiff"],
    "muscle_pain": ["body aches", "muscles hurt", "sore muscles", "whole body aches",
                    "body pain", "my body hurts", "aching all over"],
    "weakness_of_body": ["feeling weak", "body feels weak", "no energy", "no strength",
                         "feel drained", "too weak to stand"],
    "weakness_in_limbs": ["weak arms", "weak legs", "legs feel weak", "arms feel weak"],
    "headache": ["head hurts", "head is killing me", "killing head", "headache pain",
                 "pain in my head", "head pounding", "splitting head"],
    "vomiting": ["throwing up", "threw up", "cant stop throwing up", "puking",
                 "keep vomiting", "being sick"],
    "nausea": ["want to throw up", "feeling like vomiting", "queasy",
               "feeling sick to stomach", "feel like puking"],
    "diarrhea": ["loose motion", "loose motions", "runny stool", "watery stool",
                 "frequent watery stool", "upset stomach runs"],
    "stomach_pain": ["belly hurts", "tummy pain", "tummy hurts", "belly ache",
                     "stomach hurts", "stomach ache", "pain in my stomach"],
    "abdominal_pain": ["lower abdomen pain", "pain in abdomen", "abdomen hurts",
                       "cramping in my belly"],
    "cough": ["hacking cough", "keep coughing", "cant stop coughing", "dry cough",
              "chesty cough", "coughing fits"],
    "breathlessness": ["cant breathe", "hard to breathe", "trouble breathing",
                       "feeling breathless", "cant catch breath", "short of breath",
                       "out of breath", "difficulty breathing", "gasping"],
    "chest_tightness": ["chest feels tight", "tight chest", "chest is tight",
                        "band around my chest"],
    "chest_pain": ["chest hurts", "hurting in chest", "pain in my chest",
                   "chest pressure"],
    "wheezing": ["whistling breathing", "wheezing sound", "noisy breathing",
                 "whistle when i breathe"],
    "runny_nose": ["running nose", "snotty nose", "nose is runny", "nose keeps running",
                   "dripping nose"],
    "congestion": ["blocked nose", "stuffy nose", "nose is blocked", "cant breathe through nose"],
    "sore_throat": ["scratchy throat", "throat hurts", "hurting throat",
                    "painful throat", "throat is raw"],
    "high_fever": ["spiked fever", "high temperature", "burning up", "very high fever",
                   "raging fever"],
    "mild_fever": ["slight fever", "low fever", "little fever", "low grade fever"],
    "moderate_fever": ["moderate fever", "medium fever"],
    "fever": ["temperature", "feverish", "running a temperature", "hot and feverish"],
    "chills": ["shivering", "feeling cold", "cold and shivering", "chilly",
               "teeth chattering"],
    "shivering": ["shivers", "trembling with cold"],
    "sweating": ["sweating a lot", "night sweats", "sweaty", "cold sweat",
                 "drenched in sweat"],
    "dizziness": ["feeling dizzy", "feeling faint", "light headed", "lightheaded",
                  "woozy", "head spinning", "about to faint"],
    "spinning_movements": ["room is spinning", "world is spinning", "everything spins"],
    # "tired" on its own was missing: only "tired all the time" and "always
    # tired" matched, so the single most ordinary way of reporting fatigue --
    # "I feel very tired" -- extracted nothing at all. Negation is handled
    # downstream by NegEx, so the bare forms are safe to list here.
    "fatigue": ["tired all the time", "always tired", "exhausted", "worn out",
                "no stamina", "completely drained", "feel tired", "feeling tired",
                "very tired", "so tired", "really tired", "tired", "fatigued",
                "no energy", "low on energy", "lethargic", "weak and tired"],
    "malaise": ["generally unwell", "feel off", "under the weather", "not myself"],
    "itching": ["itchy skin", "skin is itchy", "itching a lot", "cant stop itching",
                "scratching constantly"],
    "skin_rash": ["rash on skin", "skin rash", "red rash", "rashes", "breaking out in rash"],
    "skin_peeling": ["skin peeling", "peeling skin", "skin flaking off"],
    "insomnia": ["cant sleep", "trouble sleeping", "hard to sleep", "lying awake",
                 "not sleeping", "cannot sleep", "unable to sleep", "no sleep",
                 "sleepless", "not able to sleep"],
    "anxiety": ["feeling anxious", "anxious all the time", "cant stop worrying",
                "on edge", "panicky"],
    "depression": ["feeling low", "feeling hopeless", "very sad", "no interest in anything"],
    "irritability": ["short tempered", "irritable", "snapping at everyone"],
    "restlessness": ["cant sit still", "restless", "fidgety"],
    "palpitations": ["heart racing", "heart pounding", "heart fluttering",
                     "can feel my heartbeat", "heart is racing",
                     "racing heart", "heart beating hard", "dhadkan tez"],
    "fast_heart_rate": ["fast pulse", "rapid heartbeat", "heart beating fast"],
    "neck_pain": ["stiff neck", "neck hurts", "hurting in neck", "pain in my neck"],
    "back_pain": ["back hurts", "lower back pain", "hurting in back", "backache"],
    "loss_of_appetite": ["not hungry", "no appetite", "dont feel like eating",
                         "cant eat", "food has no taste", "appetite is gone",
                         "lost my appetite", "do not feel like eating",
                         "bhookh nahi lagti"],
    "increased_appetite": ["always hungry", "eating more than usual", "constant hunger"],
    "weight_loss": ["losing weight", "lost weight", "clothes are loose"],
    "burning_micturition": ["burning when i pee", "burns when i urinate",
                            "painful urination", "stinging when peeing"],
    "polyuria": ["peeing a lot", "urinating frequently", "going to toilet often",
                 "keep going to the toilet", "going to the toilet often",
                 "keep going to the bathroom", "frequent urination",
                 "passing urine often", "baar baar peshab"],
    "continuous_feel_of_urine": ["always feel like peeing", "constant urge to pee"],
    "dark_urine": ["dark urine", "urine is dark", "tea coloured urine"],
    "foul_smell_of_urine": ["smelly urine", "urine smells bad"],
    "yellowing_of_eyes": ["eyes are yellow", "yellow eyes", "whites of eyes yellow",
                          "eyes look yellow", "eyes turning yellow",
                          "yellowness in eyes", "eyes have gone yellow",
                          "eyes went yellow", "eyes becoming yellow",
                          "yellowing of the eyes", "jaundiced eyes"],
    "yellowing_of_skin": ["skin is yellow", "yellow skin", "turning yellow",
                          "skin has gone yellow", "skin looks yellow"],
    "constipation": ["cant pass stool", "not passing motion", "hard stools",
                     "havent been to the toilet"],
    "bloody_stool": ["blood in stool", "blood in my motion", "bleeding when i pass stool"],
    "acidity": ["acid reflux", "heartburn", "burning in chest after eating", "sour burps"],
    "indigestion": ["cant digest", "heavy after eating", "bloated after meals"],
    "passage_of_gases": ["lot of gas", "passing gas", "flatulence"],
    "blurred_vision": ["blurry vision", "cant see clearly", "vision is blurred",
                       "things look fuzzy"],
    "watering_from_eyes": ["eyes watering", "watery eyes", "tears keep coming"],
    "continuous_sneezing": ["keep sneezing", "sneezing nonstop", "sneezing a lot",
                            "sneezing", "constant sneezing", "sneezing fits",
                            "cant stop sneezing"],
    "numbness": ["pins and needles", "cant feel my", "gone numb", "tingling"],
    "tremor": ["hands shaking", "shaky hands", "trembling"],
    "altered_sensorium": ["confused", "not thinking straight", "disoriented", "foggy headed"],
    "swelling": ["swollen", "puffy", "swelling up"],
    "face_swelling": ["face is swollen", "puffy face", "swollen face"],
    "redness": ["red and inflamed", "looks red"],
    "warmth": ["warm to touch", "feels hot to touch"],
    "tenderness": ["tender to touch", "hurts when i press"],
    "blister": ["blisters", "water filled bumps"],
    "obesity": ["overweight", "put on a lot of weight"],
    "pain": ["it hurts", "pain"],
    "pain_during_bowel_movement": ["pain when passing stool",
                                   "pain while passing stool",
                                   "hurts when passing stool",
                                   "painful bowel movement",
                                   "pain during motion",
                                   "hurts when i go to the toilet"],
    "anal_discomfort": ["discomfort around the anus", "soreness down there"],
    "bloody_stool": ["blood in stool", "blood in my motion", "bleeding in stool"],
    "face_swelling": ["swollen face", "face is swollen", "puffy face"],
    "foul_smelling_wound": ["wound smells bad", "smelly wound"],
    "loss_of_range_of_motion": ["cant move it fully", "limited movement",
                                "cannot bend it properly"],

    # ---- discriminating exposures and histories ---------------------------
    # These separate conditions whose symptom pictures overlap almost
    # entirely -- the hepatitis variants, and asthma vs COPD vs pneumonia.
    # They are asked for directly by the condition-specific follow-ups in
    # services/followup.py, but users often volunteer them unprompted, so
    # the extractor needs to recognise them in free text too.
    "recent_contaminated_food": ["ate street food", "ate outside food",
                                 "roadside food", "bad food", "stale food",
                                 "food poisoning", "ate something outside"],
    "recent_contaminated_water": ["drank unclean water", "dirty water",
                                  "unsafe water", "tap water outside",
                                  "borewell water", "flood water"],
    "unsafe_injection_exposure": ["unsterile needle", "shared needle",
                                  "reused syringe", "got an injection outside",
                                  "tattoo", "body piercing"],
    "blood_transfusion_history": ["blood transfusion", "received blood",
                                  "given blood", "transfusion earlier"],
    "known_hepatitis_b": ["i have hepatitis b", "diagnosed with hepatitis b",
                          "hep b positive", "hbsag positive",
                          "known hepatitis b"],
    "heavy_alcohol_use": ["drink a lot", "heavy drinking", "drink daily",
                          "alcohol daily", "drinking problem",
                          "lot of alcohol"],
    "pale_stools": ["pale stool", "clay coloured stool", "white stool",
                    "light coloured motion", "chalky stool"],
    "night_sweats": ["sweating at night", "night sweats", "drenched at night",
                     "sweat through the night"],
    "cough_over_two_weeks": ["cough for weeks", "coughing for a month",
                             "cough wont go away", "long standing cough",
                             "cough since weeks"],
    "productive_sputum": ["coughing up phlegm", "bringing up sputum",
                          "thick phlegm", "yellow phlegm", "green phlegm",
                          "coughing out mucus"],
    "chronic_productive_cough": ["cough with phlegm every morning",
                                 "daily cough with mucus",
                                 "cough most mornings"],
    "smoking_history": ["i smoke", "smoker", "smoking for years",
                        "cigarettes daily", "beedi", "used to smoke"],
    "night_time_symptoms": ["worse at night", "wakes me at night",
                            "cant sleep because of it", "worse when lying down"],
    "symptoms_with_triggers": ["worse with dust", "triggered by dust",
                               "worse in cold air", "after exercise",
                               "around pets", "during pollen season"],
    "reliever_not_helping": ["inhaler not working", "inhaler isnt helping",
                             "puffer not helping", "reliever not working"],
}


# ---------------------------------------------------------------------------
# Code-mixed (Hinglish / Devanagari) surface forms.
#
# Mapped to an *English phrase* rather than straight to a canonical feature,
# so the normalized text stays readable and the ordinary English matchers
# downstream do the actual entity recognition. Longest forms are matched
# first, so "tez bukhar" beats "bukhar".
# ---------------------------------------------------------------------------

CODE_MIXED: dict[str, str] = {
    # fever
    "bukhar": "fever", "bukhaar": "fever", "bukar": "fever", "jvar": "fever",
    "tez bukhar": "high fever", "tez bukhaar": "high fever",
    "halka bukhar": "mild fever", "halka bukhaar": "mild fever",
    "बुखार": "fever",
    "तेज़ बुखार": "high fever",
    # respiratory
    "khansi": "cough", "khaansi": "cough", "khasi": "cough",
    "खांसी": "cough",
    "sardi": "runny nose", "sardee": "runny nose", "zukam": "runny nose",
    "jukam": "runny nose", "zukaam": "runny nose",
    "सर्दी": "runny nose", "जुकाम": "runny nose",
    "saans phoolna": "breathlessness", "saans lene me takleef": "breathlessness",
    "sans lene me dikkat": "breathlessness", "dam phoolna": "breathlessness",
    "सांस लेने में तकलीफ": "breathlessness",
    "chheenk": "sneezing", "chhink": "sneezing", "cheenk": "sneezing",
    "gala kharab": "sore throat", "gale me dard": "sore throat",
    "gale mein dard": "sore throat", "galey me kharash": "sore throat",
    "गले में दर्द": "sore throat",
    "seene me dard": "chest pain", "chhati me dard": "chest pain",
    "सीने में दर्द": "chest pain",
    "seene me jakdan": "chest tightness",
    # head / neuro
    "sar dard": "headache", "sir dard": "headache", "sardard": "headache",
    "sirdard": "headache", "sar me dard": "headache", "sir mein dard": "headache",
    "सिरदर्द": "headache",
    "सर दर्द": "headache",
    "chakkar": "dizziness", "chakkar aana": "dizziness", "chakar": "dizziness",
    "चक्कर": "dizziness",
    "behoshi": "fainting", "ghabrahat": "anxiety", "ghabraht": "anxiety",
    "dhadkan tez": "palpitations", "dil tez dhadakna": "palpitations",
    # gastro
    "pet dard": "stomach pain", "pet me dard": "stomach pain",
    "pet mein dard": "stomach pain", "petdard": "stomach pain",
    "पेट दर्द": "stomach pain",
    "ulti": "vomiting", "ultee": "vomiting", "ulty": "vomiting", "vamit": "vomiting",
    "उल्टी": "vomiting",
    "jee michalna": "nausea", "ji michalna": "nausea", "matli": "nausea",
    "dast": "diarrhea", "dust lagna": "diarrhea", "patle dast": "diarrhea",
    "loose motion": "diarrhea", "दस्त": "diarrhea",
    "kabz": "constipation", "kabj": "constipation",
    "कब्ज": "constipation",
    "gais": "passage of gases", "pet phoolna": "indigestion",
    "khatti dakar": "acidity", "chhati me jalan": "acidity",
    "bhookh na lagna": "loss of appetite", "bhook nahi lagti": "loss of appetite",
    "bhookh kam": "loss of appetite",
    # general
    "kamzori": "weakness of body", "kamjori": "weakness of body",
    "कमजोरी": "weakness of body",
    "thakan": "fatigue", "thakaan": "fatigue", "thakavat": "fatigue",
    "थकान": "fatigue",
    "badan dard": "body aches", "sharir me dard": "body aches",
    "poore badan me dard": "body aches", "jism me dard": "body aches",
    "बदन दर्द": "body aches",
    "jodo me dard": "joint pain", "jodon mein dard": "joint pain",
    "ghutno me dard": "joint pain",
    "जोडों में दर्द": "joint pain",
    "kamar dard": "back pain", "peeth dard": "back pain",
    "gardan dard": "neck pain", "gardan me dard": "neck pain",
    "thand lagna": "chills", "thandi lagna": "chills", "sihran": "chills",
    "ठंड लगना": "chills",
    "pasina": "sweating", "paseena": "sweating", "pasina aana": "sweating",
    "पसीना": "sweating",
    "khujli": "itching", "khujali": "itching",
    "खुजली": "itching",
    "daane": "skin rash", "dane": "skin rash", "chakte": "skin rash",
    "दाने": "skin rash",
    "sujan": "swelling", "soojan": "swelling",
    "सूजन": "swelling",
    "neend na aana": "insomnia", "neend nahi aati": "insomnia",
    "nind nahi aati": "insomnia",
    "peshab me jalan": "burning micturition", "pesaab me jalan": "burning micturition",
    "peshab me dard": "burning micturition",
    "baar baar peshab": "polyuria",
    "peeliya": "yellowing of eyes", "piliya": "yellowing of eyes",
    "पीलिया": "yellowing of eyes",
    "khoon": "blood", "खून": "blood",
    "aankhon se paani": "watering from eyes",
    "dard": "pain", "दर्द": "pain",
    "takleef": "discomfort", "bimar": "unwell", "bemar": "unwell",
}


# ---------------------------------------------------------------------------
# Contractions and common misspellings handled before anything else.
# ---------------------------------------------------------------------------

CONTRACTIONS: dict[str, str] = {
    "cant": "can not", "can't": "can not", "couldnt": "could not",
    "couldn't": "could not", "dont": "do not", "don't": "do not",
    "didnt": "did not", "didn't": "did not", "doesnt": "does not",
    "doesn't": "does not", "hasnt": "has not", "hasn't": "has not",
    "havent": "have not", "haven't": "have not", "isnt": "is not",
    "isn't": "is not", "wasnt": "was not", "wasn't": "was not",
    "werent": "were not", "weren't": "were not", "wont": "will not",
    "won't": "will not", "wouldnt": "would not", "wouldn't": "would not",
    "shouldnt": "should not", "shouldn't": "should not",
    "aint": "is not", "ain't": "is not", "im": "i am", "i'm": "i am",
    "ive": "i have", "i've": "i have", "id": "i would", "i'd": "i would",
    "its": "it is", "it's": "it is", "theres": "there is", "there's": "there is",
    "thats": "that is", "that's": "that is", "youre": "you are",
    "you're": "you are", "feelin": "feeling", "hurtin": "hurting",
    "coughin": "coughing", "vomitin": "vomiting", "sneezin": "sneezing",
}


# ---------------------------------------------------------------------------
# Severity and certainty modifiers -- recognized as their own entity type so
# the UI and the triage layer can use them.
# ---------------------------------------------------------------------------

SEVERITY_MODIFIERS: dict[str, tuple[str, float]] = {
    "slight": ("mild", 0.3), "slightly": ("mild", 0.3), "mild": ("mild", 0.3),
    "mildly": ("mild", 0.3), "little": ("mild", 0.3), "bit": ("mild", 0.3),
    "minor": ("mild", 0.3), "faint": ("mild", 0.3), "low": ("mild", 0.3),
    "moderate": ("moderate", 0.6), "moderately": ("moderate", 0.6),
    "quite": ("moderate", 0.6), "fairly": ("moderate", 0.6),
    "noticeable": ("moderate", 0.6),
    "severe": ("severe", 1.0), "severely": ("severe", 1.0), "bad": ("severe", 0.8),
    "badly": ("severe", 0.8), "terrible": ("severe", 1.0), "awful": ("severe", 1.0),
    "intense": ("severe", 1.0), "extreme": ("severe", 1.0), "unbearable": ("severe", 1.0),
    "excruciating": ("severe", 1.0), "worst": ("severe", 1.0), "horrible": ("severe", 1.0),
    "killing": ("severe", 1.0), "high": ("severe", 0.8), "very": ("severe", 0.8),
    "really": ("severe", 0.8), "constant": ("severe", 0.7), "nonstop": ("severe", 0.7),
    "persistent": ("severe", 0.7),
}

BODY_SITES: dict[str, str] = {
    "head": "head", "forehead": "head", "skull": "head", "temple": "head",
    "chest": "chest", "ribs": "chest", "breastbone": "chest",
    "stomach": "abdomen", "belly": "abdomen", "tummy": "abdomen",
    "abdomen": "abdomen", "gut": "abdomen",
    "throat": "throat", "neck": "neck", "back": "back", "spine": "back",
    "knee": "joint", "knees": "joint", "elbow": "joint", "joints": "joint",
    "shoulder": "joint", "wrist": "joint", "ankle": "joint", "hip": "joint",
    "eye": "eye", "eyes": "eye", "ear": "ear", "ears": "ear",
    "nose": "nose", "skin": "skin", "leg": "limb", "legs": "limb",
    "arm": "limb", "arms": "limb", "hand": "limb", "hands": "limb",
    "foot": "limb", "feet": "limb", "mouth": "mouth", "tongue": "mouth",
    "urine": "urinary", "bladder": "urinary", "heart": "cardiac",
    "muscle": "muscle", "muscles": "muscle", "body": "generalized",
}


# ---------------------------------------------------------------------------
# Words protected from spell correction: correcting them would be a false
# positive ("cold" must not become "cough", "hand" must not become "head").
# ---------------------------------------------------------------------------

PROTECTED_WORDS: frozenset[str] = frozenset("""
a an the and or but if then than that this these those there their them they
i me my mine you your yours he she it we us our ours his her hers its
is am are was were be been being do does did done have has had having
will would shall should can could may might must
no not none never nothing nobody neither nor
of in on at to from by for with without about into over under after before
since during while until through between above below up down out off again
very really quite just only also too so such much many more most less least
some any all both each few other another same own
how what when where who whom which why whether
yes yeah yep nope okay ok sure maybe perhaps
get got go goes going gone come came take took give gave make made
feel feels feeling felt think thought know knew say said tell told
start started begin began stop stopped keep kept
day days week weeks month months year years night nights morning evening
today tomorrow yesterday now soon later hour hours minute minutes time times
doctor hospital medicine medicines tablet tablets pill pills water food eat
ate eating drink drank sleep slept work working home house family mother
father brother sister child children baby wife husband friend
little bit lot lots like well better worse good bad fine okay normal
year old age male female man woman boy girl people person
please thanks thank sorry hello hi hey help
last next past first second third best most least just must fast
felt kept left lost sent went said read need used
""".split())

# Romanized Hindi function words. Without these, bounded edit distance happily
# "corrects" Hindi grammar into clinical English -- "rahi" -> "nahi",
# "hain" -> "gain" -- which corrupts the sentence before negation ever runs.
HINDI_FUNCTION_WORDS: frozenset[str] = frozenset("""
main mai mujhe mujhko muje mera meri mere hum hamara apna apne aap tum tumhe
hai hain tha thi the hoga hogi ho hua hui hue hona
raha rahi rahe rha rhi rhe
aur ya to bhi par lekin magar kya kyun kyon kaise kab kahan kaun
se me mein ko ka ki ke par pe tak liye lie vaste
nahi nahin na mat bilkul
kuch kuchh sab sabhi bahut bohot bahot jyada zyada kam thoda thodi zara
din raat subah sham shaam hafta mahina saal kal aaj abhi roz baar
lag lagta lagti lagte lagna laga lagi
kar karna karta karti kiya gaya gayi gaye
acha achha theek thik bura kharab pareshan takleef dikkat problem
haan han ji nahi arre yaar bas sirf saath wala wali waale
oar oor pura poora poore puri
""".split())

# Frequent short misspellings. Bounded edit distance will not touch words
# under four characters (too many collisions), so these are listed directly.
MISSPELLINGS: dict[str, str] = {
    "hed": "head", "ake": "ache", "akes": "aches", "pian": "pain",
    "paon": "pain", "stomac": "stomach", "stomuch": "stomach",
    "hedache": "headache", "headake": "headache", "hedake": "headache",
    "feaver": "fever", "fver": "fever", "feever": "fever",
    "diarhea": "diarrhea", "diarhoea": "diarrhea", "diaria": "diarrhea",
    "vomitting": "vomiting", "vometing": "vomiting",
    "coff": "cough", "cof": "cough", "kough": "cough",
    "sorethroat": "sore throat", "soarthroat": "sore throat",
    "nauseaus": "nauseous", "nausious": "nauseous", "noseous": "nauseous",
    "dizzyness": "dizziness", "dizy": "dizzy",
    "tiredness": "fatigue", "weakness": "weakness",
    "breathlessnes": "breathlessness", "bretahless": "breathless",
    "swetting": "sweating", "sweeting": "sweating",
    "itchin": "itching", "ichy": "itchy", "iching": "itching",
    "rashes": "rash", "rash": "rash",
    "chils": "chills", "shivring": "shivering",
    "constipaton": "constipation", "acidty": "acidity",
    "apetite": "appetite", "appetit": "appetite",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _read_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


@lru_cache(maxsize=1)
def canonical_symptoms() -> tuple[str, ...]:
    """The model's feature order -- the only valid canonical symptom names."""
    features = _read_json(os.path.join(_MODEL_DIR, "feature_order.json"), [])
    return tuple(features)


@lru_cache(maxsize=1)
def clinical_synonyms() -> dict[str, list[str]]:
    """Curated clinical synonyms shipped with the dataset."""
    return _read_json(os.path.join(_DATA_DIR, "symptom_vocab.json"), {})


# Explicit winners where two canonical features both claim a phrase. The
# dataset synonyms and the colloquial aliases disagree in a few places, and
# resolution order should not decide a clinical distinction: "heart racing"
# is the subjective sensation (palpitations), whereas fast_heart_rate is the
# measured sign.
PHRASE_OVERRIDES: dict[str, str] = {
    "heart racing": "palpitations",
    "heart is racing": "palpitations",
    "racing heart": "palpitations",
    "heart pounding": "palpitations",
    "stomach ache": "stomach_pain",
    "stomach hurts": "stomach_pain",
    "tummy pain": "stomach_pain",
    "loose motion": "diarrhea",
    "loose motions": "diarrhea",
}


def _expand_contractions(phrase: str) -> str:
    """
    Rewrite a phrase the way the normalizer will rewrite user text.

    This matters more than it looks: the normalizer expands "cant" to
    "can not" before entity recognition runs, so an alias stored as
    "cant sleep" can never match the normalized text. Registering the
    expanded form alongside the original keeps the two in step.
    """
    words = phrase.split()
    out: list[str] = []
    for word in words:
        out.append(CONTRACTIONS.get(word, word))
    return " ".join(" ".join(out).split())


@lru_cache(maxsize=1)
def phrase_index() -> dict[str, str]:
    """
    Every known surface phrase -> canonical symptom.

    Built from the canonical name itself, the dataset synonyms, and the
    colloquial aliases above. Keys are lowercase, space-separated. Each
    phrase is registered both verbatim and contraction-expanded, because
    normalization runs before recognition.
    """
    index: dict[str, str] = {}
    synonyms = clinical_synonyms()
    known = set(canonical_symptoms())

    def register(phrase, symptom):
        key = " ".join(str(phrase).lower().replace("_", " ").split())
        if not key:
            return
        index.setdefault(key, symptom)
        expanded = _expand_contractions(key)
        if expanded != key:
            index.setdefault(expanded, symptom)

    for symptom in canonical_symptoms():
        register(symptom.replace("_", " "), symptom)
        for variant in synonyms.get(symptom, []):
            register(variant, symptom)
        for variant in COLLOQUIAL_ALIASES.get(symptom, []):
            register(variant, symptom)

    # Aliases declared for a name that is not a model feature would silently
    # never fire; fold them in only when the target exists.
    for symptom, variants in COLLOQUIAL_ALIASES.items():
        if symptom not in known:
            continue
        for variant in variants:
            register(variant, symptom)

    # Overrides are assigned, not set-defaulted, so they win regardless of
    # the order the sources above were walked in.
    for phrase, symptom in PHRASE_OVERRIDES.items():
        if symptom not in known:
            continue
        key = " ".join(phrase.lower().split())
        index[key] = symptom
        expanded = _expand_contractions(key)
        if expanded != key:
            index[expanded] = symptom

    return index


# Lay and regional names for the conditions the model can predict. A user
# naming their own condition ("I am diabetic", "my sugar is high") is useful
# context, and with a family cue it is history rather than a symptom.
DISEASE_ALIASES: dict[str, list[str]] = {
    "Diabetes": ["diabetes", "diabetic", "sugar problem", "high sugar",
                 "blood sugar", "sugar ki bimari", "madhumeh",
                 "sugar is high", "sugar level high"],
    "Hypertension": ["hypertension", "high blood pressure", "high bp", "bp high",
                     "blood pressure", "raised bp", "bp is high"],
    "Tuberculosis": ["tuberculosis", "tb", "t b"],
    "Bronchial Asthma": ["asthma", "bronchial asthma", "dama"],
    "Asthma": ["asthma attack"],
    "Dengue": ["dengue", "dengu", "break bone fever"],
    "Malaria": ["malaria", "maleria"],
    "Typhoid": ["typhoid", "tifoid", "motijhara"],
    "Jaundice": ["jaundice", "peeliya", "piliya"],
    "Migraine": ["migraine", "migrain", "adhkapari"],
    "Heart attack": ["heart attack", "cardiac arrest", "dil ka daura"],
    "Common Cold": ["common cold", "cold and flu", "flu", "influenza"],
    "Pneumonia": ["pneumonia", "nimonia", "chest infection"],
    "Gastroenteritis": ["gastroenteritis", "stomach infection", "food poisoning",
                        "stomach flu"],
    "GERD": ["gerd", "acid reflux disease", "reflux"],
    "Arthritis": ["arthritis", "gathiya", "joint disease"],
    "Gout": ["gout"],
    "Anxiety": ["anxiety disorder", "panic disorder", "panic attacks"],
    "Depression": ["depression", "clinical depression"],
    "Urinary tract infection": ["uti", "urine infection", "urinary infection",
                                "urinary tract infection"],
    "Psoriasis": ["psoriasis"],
    "Chicken pox": ["chicken pox", "chickenpox", "chhoti mata"],
    "Fungal infection": ["fungal infection", "ringworm", "daad"],
    "Allergy": ["allergy", "allergies", "allergic reaction"],
    "Hepatitis A": ["hepatitis a"], "Hepatitis B": ["hepatitis b"],
    "Hepatitis C": ["hepatitis c"], "Hepatitis D": ["hepatitis d"],
    "Hepatitis E": ["hepatitis e"],
    "COPD": ["copd", "chronic obstructive pulmonary disease"],
    "Dimorphic hemmorhoids(piles)": ["piles", "haemorrhoids", "hemorrhoids", "bawasir"],
    "Cervical spondylosis": ["cervical spondylosis", "cervical", "spondylosis"],
    "Vertigo": ["vertigo"],
    "Paralysis (brain hemorrhage)": ["paralysis", "stroke", "brain hemorrhage", "lakwa"],
    "AIDS": ["aids", "hiv"],
    "Peptic ulcer diseae": ["peptic ulcer", "stomach ulcer", "ulcer"],
    "Impetigo": ["impetigo"],
    "Diabetic foot": ["diabetic foot", "foot ulcer"],
    "Chronic cholestasis": ["cholestasis"],
    "Drug Reaction": ["drug reaction", "medicine reaction", "side effect"],
    "Alcoholic hepatitis": ["alcoholic hepatitis", "liver damage"],
}


@lru_cache(maxsize=1)
def disease_classes() -> tuple[str, ...]:
    return tuple(_read_json(os.path.join(_MODEL_DIR, "label_classes.json"), []))


@lru_cache(maxsize=1)
def disease_index() -> dict[str, str]:
    """Surface phrase -> canonical disease label, for the DISEASE entity type."""
    index: dict[str, str] = {}
    known = set(disease_classes())
    for disease in disease_classes():
        index.setdefault(" ".join(disease.lower().split()), disease)
    for disease, aliases in DISEASE_ALIASES.items():
        if disease not in known:
            continue
        for alias in aliases:
            index.setdefault(" ".join(alias.lower().split()), disease)
    return index


@lru_cache(maxsize=1)
def spell_vocabulary() -> tuple[str, ...]:
    """
    Single words a misspelling may be corrected to.

    Restricted to words that appear in the medical phrase index, so spell
    correction can only ever pull a token *towards* clinical vocabulary.
    """
    words: set[str] = set()
    for phrase in phrase_index():
        for word in phrase.split():
            if len(word) >= 4 and word not in PROTECTED_WORDS:
                words.add(word)
    # Code-mixed words are matched verbatim by the normalizer's own pass, so
    # they only need to be correction targets to absorb typos. Short ones are
    # excluded: "dast" sits one edit from "last", "ulti" from "until".
    for phrase in CODE_MIXED:
        for word in phrase.split():
            if len(word) >= 6:
                words.add(word)
    # A Hindi function word must never be a correction target either, or
    # "hai" pulls English tokens into Hindi grammar.
    words -= HINDI_FUNCTION_WORDS
    return tuple(sorted(words))


def is_protected(word: str) -> bool:
    """
    True when `word` must be left exactly as the user typed it.

    Body-site names are included: they are ordinary words the user means
    literally, and they sit one edit away from clinical terms ("hand" ->
    "band", "ear" -> "fear").
    """
    return (
        word in PROTECTED_WORDS
        or word in HINDI_FUNCTION_WORDS
        or word in BODY_SITES
    )


@lru_cache(maxsize=1)
def code_mixed_phrases() -> tuple[tuple[str, str], ...]:
    """Code-mixed phrases sorted longest-first, for greedy longest matching."""
    items = sorted(CODE_MIXED.items(), key=lambda kv: -len(kv[0].split()))
    return tuple(items)


def max_phrase_length() -> int:
    """Longest known phrase in tokens -- bounds the n-gram sweep in the NER."""
    lengths = [len(p.split()) for p in phrase_index()]
    lengths += [len(p.split()) for p in CODE_MIXED]
    return max(lengths) if lengths else 1
