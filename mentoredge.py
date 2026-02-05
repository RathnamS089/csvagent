
import pandas as pd
import random
topics = {
    "AI": [
        "Computer Vision",
        "Natural Language Processing",
        "Large Language Models",
        "Reinforcement Learning"
    ],
    "Cybersecurity": [
        "Network Security",
        "Cryptography",
        "Ethical Hacking"
    ],
    "Healthcare": [
        "Bioinformatics",
        "Medical Imaging",
        "Genomics"
    ]
}
colleges = ["MIT BLR", "IITM", "NITK", "BMSCE", "PESU", "IIST"]
names_dict={
    "Aanya":["Sharma", "Gupta", "Verma", "Rao", "Nair", "Patel", "Singh", "Reddy", "Das", "Kumar"],
    "Rohit":["Sharma", "Gupta", "Verma", "Rao", "Nair", "Patel", "Singh", "Reddy", "Das", "Kumar"],
    "Diya":["Sharma", "Gupta", "Verma", "Rao", "Nair", "Patel", "Singh", "Reddy", "Das", "Kumar"],
    "Rahul":["Sharma", "Gupta", "Verma", "Rao", "Nair", "Patel", "Singh", "Reddy", "Das", "Kumar"],
    "Sneha":["Sharma", "Gupta", "Verma", "Rao", "Nair", "Patel", "Singh", "Reddy", "Das", "Kumar"],
    "Vikram":["Sharma", "Gupta", "Verma", "Rao", "Nair", "Patel", "Singh", "Reddy", "Das", "Kumar"],
    "Priya":["Sharma", "Gupta", "Verma", "Rao", "Nair", "Patel", "Singh", "Reddy", "Das", "Kumar"],
    "Amit":["Sharma", "Gupta", "Verma", "Rao", "Nair", "Patel", "Singh", "Reddy", "Das", "Kumar"],
    "Meera":["Sharma", "Gupta", "Verma", "Rao", "Nair", "Patel", "Singh", "Reddy", "Das", "Kumar"],
    "Arjun":["Sharma", "Gupta", "Verma", "Rao", "Nair", "Patel", "Singh", "Reddy", "Das", "Kumar"]
}
def generate_profiles(num_profiles,role_name):
  data=[]
  for i in range(num_profiles):
    chosen_domain=random.choice(list(topics.keys()))
    chosen_subdomain=random.choice(topics[chosen_domain])
    name=random.choice(list(names_dict.keys()))
    last_name=random.choice(names_dict[name])
    profile={
        "name":f"{name} {last_name}",
        "college":random.choice(colleges),
        "research_domain":chosen_domain,
        "subdomain":chosen_subdomain
    }
    data.append(profile)
  return pd.DataFrame(data)
mentees_df=generate_profiles(50," Mentee")
mentor_df=generate_profiles(30,"Mentor")
mentees_df.to_csv("mentees.csv", index=False)
mentor_df.to_csv("mentors.csv", index=False)
print("Data generation complete! Checked 'mentees.csv' and 'mentors.csv'.")
print(mentees_df.head())

mentor=pd.read_csv('mentors.csv')
mentee=pd.read_csv('mentees.csv')
def calculate_match_score(mentee,mentor):
  score=0
  if mentee['research_domain']==mentor['research_domain']:
    score+=70
  if mentee['subdomain']==mentor['subdomain']:
    score+=30
  return score
results=[]
for mentee_idx,mentee_row in mentee.iterrows():
  best_mentor=None
  best_score=-1
  for mentor_idx,mentor_row in mentor.iterrows():
    current_score=calculate_match_score(mentee_row,mentor_row)
    if(current_score>best_score):
      best_score=current_score
      best_mentor=mentor_row
  if best_mentor is not None:
        results.append({
            "Mentee": mentee_row["name"],
            "Matched Mentor": best_mentor["name"],
            "Score": best_score,
            "Domain": mentee_row['research_domain']
        })
results_df = pd.DataFrame(results)
test_mentee = {
    "name": "Test Student",
    "research_domain": "AI",
    "subdomain": "Underwater Basket Weaving"
}
test_mentor = {
    "name": "Test Prof",
    "research_domain": "AI",
    "subdomain": "Computer Vision"
}
print(calculate_match_score(test_mentee, test_mentor))

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('all-MiniLM-L6-v2')
mentee_texts = mentees_df['research_domain'] + ": " + mentees_df['subdomain']
mentor_texts = mentor_df['research_domain'] + ": " + mentor_df['subdomain']
mentor_new=pd.concat([mentor,mentor],ignore_index=True)
mentor_new_texts=mentor_new['research_domain']+" :"+mentor_new['subdomain']
mentee_vector=model.encode(mentee_texts)
mentor_vector=model.encode(mentor_new_texts)
score=cosine_similarity(mentee_vector,mentor_vector)
print(score.shape)
import numpy as np
best_mentor_indices=np.argmax(score,axis=1)
mentee['matched_mentor']=mentor.iloc[best_mentor_indices]['name'].values
print(mentee[['name', 'research_domain', 'matched_mentor']].head())

from scipy.optimize import linear_sum_assignment
cost_matrix = -score
mentee_ind, mentor_slot_ind = linear_sum_assignment(cost_matrix)
mentees_df['matched_slot_id'] = mentor_slot_ind
print(mentees_df[['name', 'matched_slot_id']].head())
mentees_df['matched_mentor'] = mentor_new.iloc[mentees_df['matched_slot_id']]['name'].values
print("Fair Matches (Max 2 students per mentor):")
print(mentees_df[['name', 'research_domain', 'matched_mentor']].head())

import networkx as nx
import matplotlib.pyplot as plt
G = nx.Graph()
for mid,mrow in mentees_df.iterrows():
  G.add_edge(mrow['name'],mrow['matched_mentor'])
mentee_nodes=mentees_df['name'].unique()
pos=nx.bipartite_layout(G,mentee_nodes)
plt.figure(figsize=(12, 8))
nx.draw(G, pos,
        with_labels=True,
        node_color=['lightgreen' if node in mentee_nodes else 'lightblue' for node in G.nodes()],
        node_size=2000,
        font_size=9)
plt.title("Mentor-Mentee Research Connections")
plt.show()

row_indices = np.arange(len(mentees_df))
mentees_df['match_confidence']=score[row_indices,mentees_df['matched_slot_id']]
print(mentees_df[['name', 'matched_mentor', 'match_confidence']].head())
print(mentees_df[mentees_df['match_confidence']<0.9])
test_batch = mentees_df.sort_values('match_confidence').head(10)
test_batch['mentor_domain'] = mentor_new.iloc[test_batch['matched_slot_id']]['research_domain'].values
print("--- Quality Audit: Lowest Confidence Matches ---")
print(test_batch[['name', 'research_domain', 'matched_mentor', 'mentor_domain', 'match_confidence']])

results=[]
for idx,row in mentees_df.iterrows():
  mentor_info=mentor_new.iloc[row['matched_slot_id']]
  score=0
  reason_parts=[]
  if row['research_domain'] == mentor_info['research_domain']:
        score += 70
        reason_parts.append(f"Matching Domain ({row['research_domain']})")
  if row['subdomain'] == mentor_info['subdomain']:
        score += 30
        reason_parts.append(f"Matching Domain ({row['subdomain']})")
  if score==0:
    score=row['match_confidence']*100
    reason_parts.append("High semantic similarity")
  results.append({
        'mentee_name': row['name'],
        'matched_mentor_name': row['matched_mentor'],
        'match_reason': " & ".join(reason_parts),
        'confidence_score': round(score, 2)
    })
final_output = pd.DataFrame(results)
final_output.to_csv('final_assignments.csv', index=False)
print(final_output.head())
final_output.to_json('final_assignments.json', orient='records', indent=4)

from flask import Flask,jsonify
from flask_ngrok import run_with_ngrok
app=Flask(__name__)
@app.route('/match',methods=['GET'])
@app.route('/match', methods=['GET'])
def get_matches():
    results = []
    for idx, row in mentees_df.iterrows():
        mentor_info = mentor_new.iloc[row['matched_slot_id']]
        score = 0
        reason_parts = []
        if row['research_domain'] == mentor_info['research_domain']:
            score += 70
            reason_parts.append(f"Matching Domain ({row['research_domain']})")
        if row['sub_domain'] == mentor_info['sub_domain']:
            score += 30
            reason_parts.append(f"Matching Sub-domain ({row['sub_domain']})")
        if score == 0:
            score = row['match_confidence'] * 100
            reason_parts.append("High semantic similarity")
        results.append({
            'mentee_name': row['name'],
            'matched_mentor_name': row['matched_mentor'],
            'match_reason': " & ".join(reason_parts),
            'confidence_score': round(score, 2)
        })
    return jsonify(results)
app = Flask(__name__)
# This prints the external URL for port 5000
print("Your API is live at:")
print(output.eval_js("google.colab.kernel.proxyPort(5000)") + "match")
app.run(host='0.0.0.0', port=5000)
run_with_ngrok(app)