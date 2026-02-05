csvagent: AI-Powered Mentor Matching

csvagent is an intelligent matching engine designed to bridge the gap between mentees and mentors. By leveraging Natural Language Processing (NLP), it moves beyond simple keyword searches to understand the true semantic meaning behind research interests and professional expertise.
Key Features
Semantic Intelligence : Uses Sentence-Transformers to generate high-dimensional vector embeddings of user profiles.
Ranked Comparisons : Instead of a single match, the system calculates a similarity matrix and identifies the Top 3 most compatible mentors for every student.
Mathematical Precision : Employs Cosine Similarity to measure the distance between research domains, ensuring the highest possible accuracy.
Developer Friendly : Includes a Flask API endpoint (/match) that returns matching results in a clean JSON format.

Tech StackComponent
TechnologyLanguagePython 3.9+ 
AI/MLsentence-transformers, scikit-learn 
Data Handlingpandas, numpy 
Web Framework Flask

Clone the Repository:
git clone https://github.com/RathnamS089/csvagent.git
cd csvagent

Dependencies
pip install -r requirements.txt

Run the API
python mentoredge.py
