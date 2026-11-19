"""
Curated per-condition facts that cannot be derived from anything already in
the repo.

The rest of a fact sheet is assembled from existing project data:

    overview / typical_course / when_to_see_doctor  <- rag/knowledge_base/*.md
    common_symptoms                                 <- ml/generate_data.py profiles
    urgency                                         <- backend/app/services/severity.py
    diet / ayurvedic_lifestyle                      <- rag/ayurveda_kb.csv

so this file holds only the fields that needed to be written: lay aliases
(including Hinglish), causes, risk factors, contagiousness, complications,
condition-specific red flags, questions to ask a doctor, and the general
names of tests a doctor may order.

Editorial rules applied throughout:
  * Conservative, widely accepted public-health content only.
  * No drug names, no doses, no brands, no treatment schedules.
  * `common_tests` are general category names a patient may hear, never a
    recommendation to go and get them.
  * `sources` names a body (WHO / NHS / CDC / MedlinePlus / ICMR) only where
    that body publishes a well-known public page on the condition. Anywhere
    there was doubt it says "general medical knowledge - needs verification"
    rather than implying a citation that was not checked.
"""

NEEDS_VERIFICATION = "general medical knowledge - needs verification"

# key -> dict. Keys are the exact label strings in ml/model/label_classes.json.
PROFILES: dict[str, dict] = {

    "AIDS": dict(
        aliases=["hiv", "hiv/aids", "retroviral disease", "एचआईवी"],
        causes="Advanced infection with the human immunodeficiency virus (HIV), which damages the immune system over time.",
        risk_factors=["Unprotected sex", "Sharing needles", "Unscreened blood products", "Mother-to-child transmission"],
        is_contagious=True,
        complications=["Opportunistic infections", "Certain cancers", "Severe weight loss", "Neurological problems"],
        red_flags=["High fever that will not settle", "Breathlessness", "Severe persistent diarrhoea", "Confusion or fits"],
        ask_doctor=["Should I be tested, and how is the test done?", "What do my CD4 count and viral load mean?", "How do I protect my partner?", "Which vaccinations do I need?"],
        tests=["HIV antibody/antigen test", "CD4 count", "Viral load"],
        sources=["WHO", "CDC"],
    ),

    "Alcoholic hepatitis": dict(
        aliases=["alcoholic liver disease", "liver damage from alcohol", "sharab se liver"],
        causes="Inflammation of the liver caused by sustained heavy alcohol use.",
        risk_factors=["Long-term heavy drinking", "Binge drinking", "Existing liver disease", "Poor nutrition"],
        is_contagious=False,
        complications=["Liver scarring (cirrhosis)", "Fluid in the abdomen", "Bleeding problems", "Liver failure"],
        red_flags=["Vomiting blood", "Black tarry stools", "Deep yellowing of the eyes", "Confusion or drowsiness", "Swollen abdomen"],
        ask_doctor=["How much liver damage is there?", "What support is available to stop drinking?", "Which foods and medicines should I avoid?", "How often should liver function be rechecked?"],
        tests=["Liver function tests", "Abdominal ultrasound", "Full blood count"],
        sources=["NHS", "MedlinePlus"],
    ),

    "Allergy": dict(
        aliases=["allergic reaction", "allergies", "allergic rhinitis", "एलर्जी"],
        causes="An immune reaction to a substance that is harmless to most people, such as pollen, dust mites, certain foods or medicines.",
        risk_factors=["Family history of allergy", "Asthma or eczema", "Repeated exposure to a trigger"],
        is_contagious=False,
        complications=["Asthma flare-ups", "Sinus infections", "Rarely, a severe whole-body reaction (anaphylaxis)"],
        red_flags=["Swelling of the lips, tongue or throat", "Difficulty breathing or noisy breathing", "Feeling faint or collapsing", "Widespread hives with vomiting"],
        ask_doctor=["How do I identify my trigger?", "Is allergy testing worthwhile for me?", "What should I do if a reaction is severe?", "Will this change over time?"],
        tests=["Skin prick test", "Specific IgE blood test"],
        sources=["NHS", "WHO"],
    ),

    "Anxiety": dict(
        aliases=["anxiety disorder", "panic", "panic attacks", "ghabrahat", "बेचैनी"],
        causes="A mix of stress, life circumstances, temperament and biological factors; often no single cause can be identified.",
        risk_factors=["Ongoing stress", "Family history", "Trauma", "Some medical conditions", "Caffeine and stimulants"],
        is_contagious=False,
        complications=["Sleep problems", "Low mood", "Avoiding daily activities", "Physical symptoms such as headaches"],
        red_flags=["Thoughts of harming yourself", "Chest pain you have not had assessed", "Fainting", "Being unable to function day to day"],
        ask_doctor=["Which talking therapies are available locally?", "Could a physical condition be contributing?", "What can I do during a panic attack?", "When should treatment be reviewed?"],
        tests=["Clinical assessment", "Thyroid function tests to exclude other causes"],
        sources=["NHS", "WHO"],
    ),

    "Arthritis": dict(
        aliases=["joint pain disorder", "osteoarthritis", "gathiya", "जोड़ों का दर्द"],
        causes="Inflammation or wear of the joints; the common forms are age-related joint wear and inflammatory (autoimmune) arthritis.",
        risk_factors=["Older age", "Previous joint injury", "Excess weight", "Family history", "Repetitive joint use"],
        is_contagious=False,
        complications=["Reduced mobility", "Joint deformity", "Muscle weakness around the joint"],
        red_flags=["A hot, very swollen joint with fever", "Sudden inability to use the joint", "Unexplained weight loss with joint pain"],
        ask_doctor=["Which type of arthritis is this?", "What exercise is safe for me?", "Will physiotherapy help?", "What can I do to protect the joint long term?"],
        tests=["Joint X-ray", "Inflammatory markers (ESR/CRP)", "Rheumatoid factor"],
        sources=["NHS", "MedlinePlus"],
    ),

    "Asthma": dict(
        aliases=["asthma", "dama", "साँस की बीमारी", "reactive airway disease"],
        causes="Long-term inflammation and narrowing of the airways, which tighten in response to triggers.",
        risk_factors=["Family history", "Allergy or eczema", "Smoke and air pollution", "Occupational dusts"],
        is_contagious=False,
        complications=["Severe attacks", "Disturbed sleep", "Reduced activity", "Long-term airway changes"],
        red_flags=["Too breathless to finish a sentence", "Reliever inhaler not working", "Lips or face turning blue", "Exhaustion from the effort of breathing"],
        ask_doctor=["What are my personal triggers?", "Am I using my inhaler correctly?", "Do I need a written asthma action plan?", "How do I know an attack is severe?"],
        tests=["Spirometry (lung function)", "Peak flow monitoring"],
        sources=["WHO", "NHS"],
    ),

    "Bronchial Asthma": dict(
        aliases=["bronchial asthma", "asthma attack", "acute asthma", "dama ka daura"],
        causes="An acute flare of asthma in which the airways narrow sharply, often triggered by infection, allergens or irritants.",
        risk_factors=["Known asthma", "Recent chest infection", "Allergen or smoke exposure", "Missed preventer treatment"],
        is_contagious=False,
        complications=["Respiratory failure in severe attacks", "Repeated hospital visits"],
        red_flags=["Too breathless to speak", "Reliever not helping", "Blue lips", "Drowsiness during an attack"],
        ask_doctor=["What should I do at the first sign of a flare?", "When should I seek emergency care?", "Should my regular treatment be stepped up?"],
        tests=["Peak flow", "Oxygen saturation", "Chest X-ray if infection is suspected"],
        sources=["NHS", "WHO"],
    ),

    "COPD": dict(
        aliases=["copd", "chronic obstructive pulmonary disease", "emphysema", "chronic bronchitis", "smoker's lung"],
        causes="Long-term damage to the airways and lung tissue, most often from smoking or prolonged exposure to smoke and dust.",
        risk_factors=["Smoking", "Biomass or cooking-smoke exposure", "Occupational dust and fumes", "Repeated childhood chest infections"],
        is_contagious=False,
        complications=["Frequent chest infections", "Low blood oxygen", "Heart strain", "Reduced exercise capacity"],
        red_flags=["Sudden worsening breathlessness", "Confusion or drowsiness", "Blue lips", "Chest pain with breathlessness"],
        ask_doctor=["How severe is my airflow limitation?", "What help is available to stop smoking?", "Would pulmonary rehabilitation help?", "Which vaccinations should I have?"],
        tests=["Spirometry", "Chest X-ray", "Oxygen saturation"],
        sources=["WHO", "NHS"],
    ),

    "Cervical spondylosis": dict(
        aliases=["neck spondylosis", "cervical spine arthritis", "neck arthritis", "गर्दन का दर्द"],
        causes="Age-related wear of the discs and joints of the neck.",
        risk_factors=["Older age", "Prolonged poor posture", "Previous neck injury", "Heavy lifting"],
        is_contagious=False,
        complications=["Persistent neck pain", "Nerve compression causing arm symptoms", "Reduced neck movement"],
        red_flags=["Weakness or numbness in the arms or legs", "Problems with balance or walking", "Loss of bladder or bowel control", "Neck pain after a fall"],
        ask_doctor=["Do my symptoms suggest nerve involvement?", "Which exercises are safe?", "How should I set up my desk and pillow?", "Is imaging needed?"],
        tests=["Neck X-ray", "MRI if nerve compression is suspected"],
        sources=["NHS", "MedlinePlus"],
    ),

    "Chicken pox": dict(
        aliases=["chickenpox", "varicella", "chhoti mata", "छोटी माता"],
        causes="Infection with the varicella-zoster virus, spread by droplets and by contact with the blisters.",
        risk_factors=["Not previously infected or vaccinated", "Close contact with a case", "Childhood age group"],
        is_contagious=True,
        complications=["Skin infection where spots are scratched", "Chest infection", "More severe illness in adults, pregnancy and weakened immunity"],
        red_flags=["Difficulty breathing", "Spots becoming hot, red and spreading", "Severe headache with neck stiffness", "Drowsiness or confusion", "Chickenpox in pregnancy"],
        ask_doctor=["How long am I infectious?", "How do I relieve the itching safely?", "Who around me is at risk?", "When can I return to school or work?"],
        tests=["Usually diagnosed by appearance", "Swab of a blister if confirmation is needed"],
        sources=["NHS", "CDC"],
    ),

    "Chronic cholestasis": dict(
        aliases=["cholestasis", "bile flow obstruction", "blocked bile duct"],
        causes="Reduced or blocked flow of bile from the liver, which may follow duct disease, gallstones or certain medicines.",
        risk_factors=["Gallstones", "Liver disease", "Some medicines", "Pregnancy in some cases"],
        is_contagious=False,
        complications=["Poor absorption of fats and vitamins", "Liver scarring", "Persistent severe itching"],
        red_flags=["Fever with yellowing of the eyes", "Severe upper abdominal pain", "Pale stools with dark urine", "Confusion"],
        ask_doctor=["What is blocking the bile flow?", "Could any of my medicines be responsible?", "What can be done about the itching?", "Which vitamins should be monitored?"],
        tests=["Liver function tests", "Abdominal ultrasound", "Bile duct imaging"],
        sources=[NEEDS_VERIFICATION],
    ),

    "Common Cold": dict(
        aliases=["cold", "common cold", "sardi", "sardi jukam", "जुकाम"],
        causes="Viral infection of the nose and throat, most often by rhinoviruses, spread by droplets and contaminated hands.",
        risk_factors=["Close contact with someone infected", "Crowded indoor spaces", "Cold season", "Young children at home"],
        is_contagious=True,
        complications=["Ear infection", "Sinus infection", "Flare-up of asthma"],
        red_flags=["Breathlessness or chest pain", "Fever above 102F lasting more than three days", "Severe headache with neck stiffness", "Symptoms worsening after initially improving"],
        ask_doctor=["How long should this last?", "How do I avoid passing it on?", "Which symptoms mean it is no longer just a cold?"],
        tests=["Usually none; diagnosed clinically"],
        sources=["NHS", "CDC"],
    ),

    "Dengue": dict(
        aliases=["dengue", "dengue fever", "breakbone fever", "डेंगू", "haddi tod bukhar"],
        causes="Infection with dengue virus, spread by the bite of infected Aedes mosquitoes which bite mainly during the day.",
        risk_factors=["Living in or travelling to an area with dengue", "Standing water near the home", "Previous dengue infection"],
        is_contagious=False,
        complications=["Bleeding", "Dangerous drop in blood pressure", "Low platelet count", "Severe dengue, which is a medical emergency"],
        red_flags=["Severe abdominal pain", "Persistent vomiting", "Bleeding gums or nose", "Blood in vomit or stool", "Cold clammy skin or extreme restlessness"],
        ask_doctor=["Do I need a blood test, and when?", "What warning signs should I watch for at home?", "Which pain relief is safe for me?", "How much fluid should I drink?"],
        tests=["Dengue NS1 antigen", "Dengue antibody test", "Full blood count with platelets"],
        sources=["WHO", "ICMR"],
    ),

    "Depression": dict(
        aliases=["depression", "low mood", "clinical depression", "udaasi", "अवसाद"],
        causes="A combination of biological, psychological and social factors; it is an illness, not a personal weakness.",
        risk_factors=["Family history", "Major life stress or loss", "Chronic illness", "Isolation", "Substance use"],
        is_contagious=False,
        complications=["Difficulty working or studying", "Relationship strain", "Physical health decline", "Risk of self-harm"],
        red_flags=["Thoughts of suicide or self-harm", "Being unable to care for yourself", "Hearing or seeing things others do not", "Not eating or drinking"],
        ask_doctor=["What treatment options do I have?", "How long before I might feel better?", "What should I do on a bad day?", "Who can I contact in a crisis?"],
        tests=["Clinical assessment", "Blood tests to exclude other causes such as thyroid or anaemia"],
        sources=["WHO", "NHS"],
    ),

    "Diabetes": dict(
        aliases=["diabetes", "sugar", "sugar ki bimari", "blood sugar", "मधुमेह", "diabetes mellitus", "high sugar"],
        causes="The body either does not make enough insulin or cannot use it properly, so glucose builds up in the blood.",
        risk_factors=["Family history", "Excess weight", "Physical inactivity", "Older age", "Gestational diabetes in the past"],
        is_contagious=False,
        complications=["Eye, kidney and nerve damage", "Foot ulcers", "Heart and blood vessel disease"],
        red_flags=["Very high or very low sugar readings", "Drowsiness or confusion", "Fruity-smelling breath with vomiting", "A foot wound that is not healing"],
        ask_doctor=["What are my target sugar levels?", "How often should HbA1c be checked?", "Which screening do I need for eyes, kidneys and feet?", "How should I handle sick days?"],
        tests=["Fasting blood glucose", "HbA1c", "Urine protein", "Eye and foot checks"],
        sources=["WHO", "ICMR"],
    ),

    "Diabetic foot": dict(
        aliases=["diabetic foot ulcer", "foot ulcer", "diabetic wound", "pair ka ghaav"],
        causes="Nerve damage and reduced blood supply from long-standing diabetes, which lets small injuries become ulcers.",
        risk_factors=["Long-standing or poorly controlled diabetes", "Loss of sensation in the feet", "Poor circulation", "Ill-fitting footwear", "Previous ulcer"],
        is_contagious=False,
        complications=["Deep infection", "Bone infection", "Tissue death", "Amputation if neglected"],
        red_flags=["Spreading redness or swelling", "Foul-smelling discharge", "Fever with a foot wound", "Black tissue", "Sudden severe foot pain"],
        ask_doctor=["How should this wound be dressed?", "Do I need to stay off the foot?", "Is my circulation adequate?", "How often should my feet be checked?"],
        tests=["Wound assessment", "Foot pulses and sensation testing", "Wound swab", "X-ray if bone involvement is suspected"],
        sources=["NHS", "WHO"],
    ),

    "Dimorphic hemmorhoids(piles)": dict(
        aliases=["piles", "haemorrhoids", "hemorrhoids", "bawaseer", "बवासीर", "pile"],
        causes="Swollen blood vessels in and around the back passage, commonly from straining, constipation or prolonged sitting.",
        risk_factors=["Constipation and straining", "Low-fibre diet", "Pregnancy", "Prolonged sitting", "Heavy lifting"],
        is_contagious=False,
        complications=["Ongoing bleeding and anaemia", "Painful clotted pile", "Prolapse"],
        red_flags=["Heavy or continuous bleeding", "Dizziness or fainting with bleeding", "Severe anal pain", "A change in bowel habit lasting weeks", "Unexplained weight loss"],
        ask_doctor=["Is the bleeding definitely from piles?", "Do I need an examination to rule out other causes?", "How can I soften my stools?", "When is a procedure considered?"],
        tests=["Clinical examination", "Proctoscopy", "Colonoscopy if other causes need excluding"],
        sources=["NHS", "MedlinePlus"],
    ),

    "Drug Reaction": dict(
        aliases=["drug allergy", "medicine reaction", "adverse drug reaction", "dawa se reaction"],
        causes="An unwanted immune or toxic response to a medicine.",
        risk_factors=["Previous drug reaction", "Multiple medicines", "Certain antibiotics and anticonvulsants", "Known allergies"],
        is_contagious=False,
        complications=["Severe skin reactions", "Liver or kidney injury", "Anaphylaxis"],
        red_flags=["Swelling of the face, lips or tongue", "Difficulty breathing", "Blistering or peeling skin", "Skin pain with fever", "Sores in the mouth or eyes"],
        ask_doctor=["Which medicine is most likely responsible?", "Should I stop it, and what do I use instead?", "Should this be recorded as an allergy?", "What should I avoid in future?"],
        tests=["Clinical assessment", "Blood tests for organ involvement", "Allergy referral in selected cases"],
        sources=["NHS", "MedlinePlus"],
    ),

    "Fungal infection": dict(
        aliases=["fungal infection", "ringworm", "dad", "daad", "दाद", "tinea", "skin fungus"],
        causes="Infection of the skin, nails or scalp by fungi, favoured by warmth, sweat and moisture.",
        risk_factors=["Hot humid weather", "Sweating and tight clothing", "Sharing towels or combs", "Diabetes", "Weakened immunity"],
        is_contagious=True,
        complications=["Spread to other body areas or household members", "Secondary bacterial infection", "Nail involvement"],
        red_flags=["Rapidly spreading redness with fever", "Pus and marked swelling", "Fungal infection with uncontrolled diabetes", "Scalp involvement with hair loss"],
        ask_doctor=["How long does treatment usually take?", "How do I stop it spreading at home?", "Why does it keep coming back?", "Do my nails need separate treatment?"],
        tests=["Skin scraping for microscopy", "Fungal culture"],
        sources=["NHS", "CDC"],
    ),

    "GERD": dict(
        aliases=["gerd", "acid reflux", "acidity", "heartburn", "reflux", "खट्टी डकार", "gas ki problem"],
        causes="Stomach acid flowing back into the food pipe because the valve between them does not close effectively.",
        risk_factors=["Excess weight", "Large or late meals", "Spicy or fatty food", "Smoking and alcohol", "Pregnancy"],
        is_contagious=False,
        complications=["Inflammation of the food pipe", "Narrowing", "Disturbed sleep", "Dental erosion"],
        red_flags=["Difficulty or pain on swallowing", "Vomiting blood or black stools", "Unintended weight loss", "Chest pain spreading to the arm or jaw"],
        ask_doctor=["Which foods and habits should I change first?", "Do my symptoms need an endoscopy?", "Could my chest pain be from the heart?", "How long should treatment continue?"],
        tests=["Clinical assessment", "Endoscopy where warning features are present"],
        sources=["NHS", "MedlinePlus"],
    ),

    "Gastroenteritis": dict(
        aliases=["stomach flu", "stomach infection", "food poisoning", "pet kharab", "दस्त"],
        causes="Infection of the stomach and bowel, usually viral but sometimes bacterial, often from contaminated food or water.",
        risk_factors=["Contaminated food or water", "Close contact with a case", "Poor hand hygiene", "Travel"],
        is_contagious=True,
        complications=["Dehydration", "Electrolyte disturbance", "Greater risk in infants and older adults"],
        red_flags=["Signs of dehydration such as very little urine or dizziness on standing", "Blood in the stool", "Persistent vomiting with no fluids kept down", "High fever", "Severe constant abdominal pain"],
        ask_doctor=["How do I rehydrate properly?", "When is a stool test needed?", "When can I return to work or school?", "What should I eat while recovering?"],
        tests=["Usually none", "Stool test if severe, bloody or prolonged"],
        sources=["WHO", "NHS"],
    ),

    "Gout": dict(
        aliases=["gout", "gouty arthritis", "uric acid problem", "yurik acid"],
        causes="Uric acid crystals forming in a joint, causing sudden intense inflammation.",
        risk_factors=["High uric acid levels", "Red meat and seafood", "Alcohol, especially beer", "Sugary drinks", "Kidney problems", "Family history"],
        is_contagious=False,
        complications=["Repeated attacks", "Joint damage", "Uric acid kidney stones", "Lumps of crystal under the skin"],
        red_flags=["A hot swollen joint with fever and chills", "More than one joint suddenly affected", "Inability to move the joint at all"],
        ask_doctor=["Is this gout or an infected joint?", "Which foods and drinks matter most?", "Should my uric acid be treated long term?", "How do I manage a future attack?"],
        tests=["Serum uric acid", "Joint fluid examination", "Kidney function tests"],
        sources=["NHS", "MedlinePlus"],
    ),

    "Heart attack": dict(
        aliases=["heart attack", "myocardial infarction", "mi", "cardiac arrest", "dil ka daura", "दिल का दौरा"],
        causes="Blood flow to part of the heart muscle is suddenly blocked, usually by a clot in a narrowed coronary artery.",
        risk_factors=["Smoking", "High blood pressure", "Diabetes", "High cholesterol", "Family history", "Obesity and inactivity"],
        is_contagious=False,
        complications=["Heart rhythm disturbance", "Heart failure", "Cardiac arrest", "Death without prompt treatment"],
        red_flags=["Chest pain or pressure lasting more than a few minutes", "Pain spreading to the arm, jaw, neck or back", "Breathlessness with sweating", "Nausea with chest discomfort", "Collapse"],
        ask_doctor=["Call emergency services first - ask questions afterwards.", "What caused this and how is further risk reduced?", "What activity is safe during recovery?", "Is cardiac rehabilitation available?"],
        tests=["ECG", "Cardiac troponin blood test", "Coronary angiography"],
        sources=["WHO", "NHS"],
    ),

    "Hepatitis A": dict(
        aliases=["hepatitis a", "hep a", "infectious hepatitis", "peelia", "jaundice from food"],
        causes="Infection of the liver with hepatitis A virus, spread through contaminated food or water.",
        risk_factors=["Unsafe food or water", "Poor sanitation", "Close contact with a case", "Travel to areas with poor sanitation"],
        is_contagious=True,
        complications=["Prolonged fatigue", "Rarely, severe liver inflammation"],
        red_flags=["Confusion or drowsiness", "Vomiting blood", "Deep yellowing with bruising", "Unable to keep fluids down"],
        ask_doctor=["How long will recovery take?", "How do I avoid passing it on?", "Should my household be vaccinated?", "Which medicines should I avoid?"],
        tests=["Hepatitis A antibody test", "Liver function tests"],
        sources=["WHO", "CDC"],
    ),

    "Hepatitis B": dict(
        aliases=["hepatitis b", "hep b", "hbv", "hbsag positive"],
        causes="Infection of the liver with hepatitis B virus, spread through blood and body fluids.",
        risk_factors=["Unsterile injections or procedures", "Unprotected sex", "Mother-to-child transmission", "Sharing razors or needles"],
        is_contagious=True,
        complications=["Long-term (chronic) infection", "Liver scarring", "Liver cancer in chronic cases"],
        red_flags=["Confusion or drowsiness", "Vomiting blood", "Swollen abdomen", "Deep jaundice"],
        ask_doctor=["Is my infection recent or long-standing?", "Should my family be tested and vaccinated?", "How often should my liver be monitored?", "What precautions protect others?"],
        tests=["HBsAg", "Hepatitis B viral load", "Liver function tests"],
        sources=["WHO", "CDC"],
    ),

    "Hepatitis C": dict(
        aliases=["hepatitis c", "hep c", "hcv"],
        causes="Infection of the liver with hepatitis C virus, spread mainly through blood-to-blood contact.",
        risk_factors=["Unscreened blood transfusion", "Unsterile injections", "Sharing needles", "Unsterile tattooing"],
        is_contagious=True,
        complications=["Chronic infection", "Liver scarring", "Liver cancer"],
        red_flags=["Confusion", "Vomiting blood", "Swollen abdomen", "Deep jaundice"],
        ask_doctor=["Do I have ongoing infection?", "How much liver damage is there?", "What treatment is available to me?", "How do I protect others?"],
        tests=["Hepatitis C antibody test", "HCV RNA", "Liver function tests"],
        sources=["WHO", "CDC"],
    ),

    "Hepatitis D": dict(
        aliases=["hepatitis d", "hep d", "delta hepatitis", "hdv"],
        causes="Infection with hepatitis D virus, which can only occur in someone who also has hepatitis B.",
        risk_factors=["Existing hepatitis B infection", "Blood exposure", "Unsterile injections"],
        is_contagious=True,
        complications=["More rapid liver damage than hepatitis B alone", "Liver failure", "Cirrhosis"],
        red_flags=["Confusion or drowsiness", "Vomiting blood", "Rapidly deepening jaundice", "Swollen abdomen"],
        ask_doctor=["Has my hepatitis B status been confirmed?", "How is this monitored?", "What does this mean for my liver long term?", "How do I prevent transmission?"],
        tests=["Hepatitis D antibody test", "Hepatitis B tests", "Liver function tests"],
        sources=["WHO"],
    ),

    "Hepatitis E": dict(
        aliases=["hepatitis e", "hep e", "hev"],
        causes="Infection of the liver with hepatitis E virus, spread mainly through contaminated drinking water.",
        risk_factors=["Unsafe drinking water", "Poor sanitation", "Flooding", "Pregnancy carries higher risk of severe illness"],
        is_contagious=True,
        complications=["Usually self-limiting", "Severe illness particularly in pregnancy", "Rarely liver failure"],
        red_flags=["Pregnancy with jaundice", "Confusion or drowsiness", "Vomiting blood", "Unable to keep fluids down"],
        ask_doctor=["How long will this last?", "Is my drinking water safe?", "What extra care is needed if pregnant?", "Which medicines should I avoid?"],
        tests=["Hepatitis E antibody test", "Liver function tests"],
        sources=["WHO"],
    ),

    "Hypertension": dict(
        aliases=["hypertension", "high bp", "bp", "high blood pressure", "blood pressure", "उच्च रक्तचाप", "bp high"],
        causes="Persistently raised pressure in the arteries; usually no single cause is found, though kidney and hormonal causes exist.",
        risk_factors=["Family history", "High salt intake", "Excess weight", "Inactivity", "Alcohol", "Stress", "Older age"],
        is_contagious=False,
        complications=["Stroke", "Heart attack and heart failure", "Kidney damage", "Eye damage"],
        red_flags=["Severe headache with blurred vision", "Chest pain", "Breathlessness at rest", "Weakness on one side or slurred speech", "Very high readings with symptoms"],
        ask_doctor=["What is my target blood pressure?", "How often should I measure it at home?", "Which lifestyle change will help me most?", "What should I do if a reading is very high?"],
        tests=["Repeated blood pressure measurement", "Kidney function tests", "ECG", "Cholesterol and glucose"],
        sources=["WHO", "NHS"],
    ),

    "Impetigo": dict(
        aliases=["impetigo", "skin infection with crust", "school sores"],
        causes="A contagious bacterial infection of the surface of the skin.",
        risk_factors=["Breaks in the skin such as scratches or insect bites", "Close contact and crowding", "Warm humid weather", "Existing eczema"],
        is_contagious=True,
        complications=["Spread to other sites or people", "Deeper skin infection", "Rarely kidney inflammation"],
        red_flags=["Fever with spreading redness", "Large painful blisters", "Rapid spread over hours", "Impetigo in a newborn"],
        ask_doctor=["How long am I infectious?", "How do I prevent it spreading at home?", "When can my child return to school?", "What hygiene measures help?"],
        tests=["Usually diagnosed by appearance", "Skin swab if not settling"],
        sources=["NHS", "CDC"],
    ),

    "Jaundice": dict(
        aliases=["jaundice", "peelia", "पीलिया"],
        causes="A build-up of bilirubin, which is a sign of an underlying problem in the liver, bile ducts or blood rather than a disease in itself.",
        risk_factors=["Liver infection or inflammation", "Bile duct obstruction", "Certain medicines", "Conditions that break down red blood cells"],
        is_contagious=False,
        complications=["Depends entirely on the underlying cause", "Itching", "Poor absorption of nutrients"],
        red_flags=["Confusion or drowsiness", "Fever with jaundice", "Severe abdominal pain", "Vomiting blood or black stools", "Jaundice in a newborn"],
        ask_doctor=["What is causing the jaundice?", "Which tests will identify the cause?", "Which medicines should I stop or avoid?", "What should I watch for at home?"],
        tests=["Liver function tests including bilirubin", "Abdominal ultrasound", "Hepatitis screening", "Full blood count"],
        sources=["NHS", "MedlinePlus"],
    ),

    "Malaria": dict(
        aliases=["malaria", "मलेरिया", "mosquito fever", "plasmodium"],
        causes="Infection with Plasmodium parasites transmitted by the bite of infected female Anopheles mosquitoes.",
        risk_factors=["Living in or travelling to a malaria area", "No mosquito-net use", "Standing water nearby", "Monsoon season"],
        is_contagious=False,
        complications=["Severe anaemia", "Cerebral malaria", "Organ failure", "Death if untreated"],
        red_flags=["Confusion or fits", "Very dark or minimal urine", "Difficulty breathing", "Repeated vomiting with inability to take fluids", "Yellowing of the eyes"],
        ask_doctor=["Which blood test confirms malaria?", "Which species is it?", "How soon should I feel better?", "How do I protect my household?"],
        tests=["Malaria rapid diagnostic test", "Blood smear microscopy", "Full blood count"],
        sources=["WHO", "ICMR"],
    ),

    "Migraine": dict(
        aliases=["migraine", "adha sir dard", "आधा सिर दर्द"],
        causes="A neurological condition producing recurrent headache attacks, often with sensitivity to light and sound.",
        risk_factors=["Family history", "Stress and poor sleep", "Skipped meals", "Certain foods", "Hormonal changes"],
        is_contagious=False,
        complications=["Frequent attacks affecting work and study", "Medication-overuse headache"],
        red_flags=["The worst headache of your life, coming on suddenly", "Headache with fever and neck stiffness", "Weakness, numbness or difficulty speaking", "Headache after a head injury", "New headache pattern over the age of 50"],
        ask_doctor=["What are my triggers?", "Would a headache diary help?", "When is preventive treatment considered?", "How do I avoid overusing painkillers?"],
        tests=["Usually clinical", "Brain imaging only if warning features are present"],
        sources=["WHO", "NHS"],
    ),

    "Paralysis (brain hemorrhage)": dict(
        aliases=["stroke", "brain hemorrhage", "brain haemorrhage", "paralysis", "lakwa", "लकवा", "brain stroke"],
        causes="Bleeding into or around the brain, which damages brain tissue and interrupts its blood supply.",
        risk_factors=["High blood pressure", "Smoking", "Blood-thinning medicines", "Aneurysms", "Heavy alcohol use"],
        is_contagious=False,
        complications=["Lasting weakness or speech difficulty", "Swallowing problems", "Seizures", "Death without urgent treatment"],
        red_flags=["Sudden weakness of the face, arm or leg, usually one side", "Sudden difficulty speaking or understanding", "Sudden worst-ever headache", "Sudden loss of vision or balance", "Any of these - call emergency services immediately"],
        ask_doctor=["Call emergency services first - time is critical.", "What caused the bleed?", "What rehabilitation is available?", "How is the risk of another event reduced?"],
        tests=["Urgent CT scan of the head", "MRI", "Blood pressure and clotting tests"],
        sources=["WHO", "NHS"],
    ),

    "Peptic ulcer diseae": dict(
        aliases=["peptic ulcer", "stomach ulcer", "duodenal ulcer", "ulcer", "pet ka alsar"],
        causes="A sore in the lining of the stomach or upper small intestine, commonly from H. pylori infection or prolonged use of anti-inflammatory painkillers.",
        risk_factors=["H. pylori infection", "Regular anti-inflammatory painkillers", "Smoking", "Heavy alcohol use"],
        is_contagious=False,
        complications=["Bleeding", "Perforation", "Obstruction of the stomach outlet"],
        red_flags=["Vomiting blood or material like coffee grounds", "Black tarry stools", "Sudden severe abdominal pain with a rigid abdomen", "Fainting", "Unintended weight loss"],
        ask_doctor=["Should I be tested for H. pylori?", "Which painkillers are safe for me?", "Do I need an endoscopy?", "What dietary changes help?"],
        tests=["H. pylori breath, stool or blood test", "Endoscopy"],
        sources=["NHS", "MedlinePlus"],
    ),

    "Pneumonia": dict(
        aliases=["pneumonia", "chest infection", "lung infection", "निमोनिया", "phephde ka infection"],
        causes="Infection of the lung tissue, most often bacterial or viral, causing the air sacs to fill with fluid.",
        risk_factors=["Age under 5 or over 65", "Smoking", "Chronic lung or heart disease", "Weakened immunity", "Recent viral illness"],
        is_contagious=True,
        complications=["Low blood oxygen", "Fluid around the lung", "Spread of infection to the blood", "Respiratory failure"],
        red_flags=["Rapid or laboured breathing", "Blue lips or face", "Confusion, especially in older people", "Chest pain on breathing", "Unable to complete a sentence"],
        ask_doctor=["Do I need a chest X-ray?", "Should my oxygen level be checked?", "How long before I improve?", "Which vaccinations would help prevent this?"],
        tests=["Chest X-ray", "Oxygen saturation", "Full blood count", "Sputum test in some cases"],
        sources=["WHO", "NHS"],
    ),

    "Psoriasis": dict(
        aliases=["psoriasis", "scaly skin disease", "sorasis"],
        causes="A long-term immune-mediated condition in which skin cells are produced far faster than normal.",
        risk_factors=["Family history", "Stress", "Skin injury", "Infections", "Smoking", "Some medicines"],
        is_contagious=False,
        complications=["Joint involvement (psoriatic arthritis)", "Skin infection", "Significant effect on mood and confidence"],
        red_flags=["Widespread redness covering most of the body", "Pustules with fever", "Rapidly worsening joint pain and swelling"],
        ask_doctor=["Which type of psoriasis is this?", "What triggers my flares?", "How do I care for my skin day to day?", "Should my joints be assessed?"],
        tests=["Usually diagnosed by appearance", "Skin biopsy in unclear cases"],
        sources=["NHS", "MedlinePlus"],
    ),

    "Tuberculosis": dict(
        aliases=["tb", "tuberculosis", "टीबी", "kshay rog", "lung tb"],
        causes="Infection with Mycobacterium tuberculosis, usually affecting the lungs and spread through the air when an infected person coughs.",
        risk_factors=["Close contact with an active case", "Crowded or poorly ventilated living conditions", "HIV or weakened immunity", "Diabetes", "Malnutrition", "Smoking"],
        is_contagious=True,
        complications=["Lung damage", "Spread to other organs", "Drug-resistant TB if treatment is interrupted"],
        red_flags=["Coughing up blood", "Severe breathlessness", "Chest pain with high fever", "Rapid unintended weight loss", "Confusion or neck stiffness"],
        ask_doctor=["Which tests confirm TB?", "How long is the treatment and why must it be completed?", "When do I stop being infectious?", "Should my household be screened?"],
        tests=["Sputum test for TB", "Chest X-ray", "Molecular test such as NAAT", "Tuberculin skin test"],
        sources=["WHO", "ICMR"],
    ),

    "Typhoid": dict(
        aliases=["typhoid", "typhoid fever", "enteric fever", "टाइफाइड", "motijhara"],
        causes="Infection with Salmonella Typhi, acquired from contaminated food or water.",
        risk_factors=["Unsafe food or water", "Poor sanitation", "Close contact with a case or carrier", "Travel to areas where typhoid is common"],
        is_contagious=True,
        complications=["Intestinal bleeding or perforation", "Prolonged fever", "Carrier state"],
        red_flags=["Severe abdominal pain with a rigid abdomen", "Blood in stool or black stools", "Confusion or extreme drowsiness", "Persistent vomiting"],
        ask_doctor=["Which test confirms typhoid?", "How long is the full course of treatment?", "When can I return to work, especially if I handle food?", "Should my household be checked?"],
        tests=["Blood culture", "Typhoid serology", "Full blood count"],
        sources=["WHO", "ICMR"],
    ),

    "Urinary tract infection": dict(
        aliases=["uti", "urine infection", "urinary infection", "bladder infection", "peshab me jalan", "पेशाब में जलन"],
        causes="Bacteria entering and multiplying in the urinary tract, most often the bladder.",
        risk_factors=["Female anatomy", "Sexual activity", "Not emptying the bladder fully", "Low fluid intake", "Diabetes", "Catheter use"],
        is_contagious=False,
        complications=["Kidney infection", "Recurrent infections", "Infection spreading to the blood"],
        red_flags=["Fever with flank or back pain", "Shaking chills", "Vomiting and unable to keep fluids down", "Blood in the urine", "Confusion in an older person", "UTI in pregnancy"],
        ask_doctor=["Does my urine need testing?", "How do I reduce repeat infections?", "How much should I be drinking?", "Which symptoms mean the kidneys are involved?"],
        tests=["Urine dipstick", "Urine culture"],
        sources=["NHS", "MedlinePlus"],
    ),

    "Vertigo": dict(
        aliases=["vertigo", "spinning sensation", "chakkar", "चक्कर", "bppv"],
        causes="A false sense of spinning, most commonly from a problem in the balance organs of the inner ear.",
        risk_factors=["Inner ear disorders", "Head injury", "Migraine", "Older age", "Some medicines"],
        is_contagious=False,
        complications=["Falls and injury", "Nausea and dehydration", "Avoiding activity"],
        red_flags=["Vertigo with weakness, numbness or slurred speech", "Severe headache with vertigo", "Double vision", "Difficulty walking or loss of coordination", "Vertigo after a head injury"],
        ask_doctor=["Is this coming from the ear or something else?", "Are there exercises or manoeuvres that help?", "Is it safe for me to drive?", "How do I prevent falls at home?"],
        tests=["Clinical balance examination", "Positional testing", "Hearing test", "Brain imaging if central causes are suspected"],
        sources=["NHS", "MedlinePlus"],
    ),
}
