# **1. Importing Necessary Libraries** 📚

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import tree,svm
from sklearn.ensemble import RandomForestClassifier
import pickle 
import streamlit as st

pickle_a=open("cp.pkl","rb")
regressor=pickle.load(pickle_a) # our model

# **2. Loading Dataset** 📊

df = pd.read_csv('https://raw.githubusercontent.com/Umang-19/devjam/main/public/mldata.csv')
df.head()

df['workshops'] = df['workshops'].replace(['testing'],'Testing')
df.head()

print(df.columns.unique)

n = df['Suggested Job Role'].unique()
print(len(n))

print('The shape of our training set: %s professionals and %s features'%(df.shape[0],df.shape[1]))


# **5. Feature Engineering** ⚙️

## (a) Binary Encoding for Categorical Variables

newdf = df
newdf.head(10)

cols = df[["self-learning capability?", "Extra-courses did","Taken inputs from seniors or elders", "worked in teams ever?", "Introvert"]]
for i in cols:
    print(i)
    cleanup_nums = {i: {"yes": 1, "no": 0}}
    df = df.replace(cleanup_nums)

print("\n\nList of Categorical features: \n" , df.select_dtypes(include=['object']).columns.tolist())

## (b) Number Encoding for Categorical 

mycol = df[["reading and writing skills", "memory capability score"]]
for i in mycol:
    print(i)    
    cleanup_nums = {i: {"poor": 0, "medium": 1, "excellent": 2}}
    df = df.replace(cleanup_nums)

category_cols = df[['certifications', 'workshops', 'Interested subjects', 'interested career area ', 'Type of company want to settle in?', 
                    'Interested Type of Books']]
for i in category_cols:
    df[i] = df[i].astype('category')
    df[i + "_code"] = df[i].cat.codes

print("\n\nList of Categorical features: \n" , df.select_dtypes(include=['object']).columns.tolist())

## (c) Dummy Variable Encoding

print(df['Management or Technical'].unique())
print(df['hard/smart worker'].unique())

df = pd.get_dummies(df, columns=["Management or Technical", "hard/smart worker"], prefix=["A", "B"])
df.head()

df.sort_values(by=['certifications'])

print("List of Numerical features: \n" , df.select_dtypes(include=np.number).columns.tolist())


category_cols = df[['certifications', 'workshops', 'Interested subjects', 'interested career area ', 'Type of company want to settle in?', 
                    'Interested Type of Books']]
for i in category_cols:
  print(i)

Certifi = list(df['certifications'].unique())
print(Certifi)
certi_code = list(df['certifications_code'].unique())
print(certi_code)

Workshops = list(df['workshops'].unique())
print(Workshops)
Workshops_code = list(df['workshops_code'].unique())
print(Workshops_code)

Certi_l = list(df['certifications'].unique())
certi_code = list(df['certifications_code'].unique())
C = dict(zip(Certi_l,certi_code))

Workshops = list(df['workshops'].unique())
print(Workshops)
Workshops_code = list(df['workshops_code'].unique())
print(Workshops_code)
W = dict(zip(Workshops,Workshops_code))

Interested_subjects = list(df['Interested subjects'].unique())
print(Interested_subjects)
Interested_subjects_code = list(df['Interested subjects_code'].unique())
ISC = dict(zip(Interested_subjects,Interested_subjects_code))

interested_career_area = list(df['interested career area '].unique())
print(interested_career_area)
interested_career_area_code = list(df['interested career area _code'].unique())
ICA = dict(zip(interested_career_area,interested_career_area_code))

Typeofcompany = list(df['Type of company want to settle in?'].unique())
print(Typeofcompany)
Typeofcompany_code = list(df['Type of company want to settle in?_code'].unique())
TOCO = dict(zip(Typeofcompany,Typeofcompany_code))

Interested_Books = list(df['Interested Type of Books'].unique())
print(Interested_subjects)
Interested_Books_code = list(df['Interested Type of Books_code'].unique())
IB = dict(zip(Interested_Books,Interested_Books_code))

Range_dict = {"poor": 0, "medium": 1, "excellent": 2}
print(Range_dict)


A = 'yes'
B = 'No'
col = [A,B]
for i in col:
  if(i=='yes'):
    i = 1
  print(i)


f =[]
A = 'r programming'
clms = ['r programming',0]
for i in clms:
  for key in C:
    if(i==key):
      i = C[key]
      f.append(i)
print(f)

C = dict(zip(Certifi,certi_code))
  
print(C)

import numpy as np
array = np.array([1,2,3,4])
array.reshape(-1,1)

def inputlist(Name,Contact_Number,Email_address,Logical_quotient_rating, coding_skills_rating, hackathons, public_speaking_points, self_learning_capability, 
       Extra_courses_did, Taken_inputs_from_seniors_or_elders,worked_in_teams_ever,Introvert, reading_and_writing_skills,
       memory_capability_score, smart_or_hard_work, Magement_or_Techinical,
       Interested_subjects, Interested_Type_of_Books,certifications, workshops, Type_of_company_want_to_settle_in, interested_career_area ):
  #1,1,1,1,'Yes','Yes''Yes''Yes''Yes',"poor","poor","Smart worker", "Management","programming","Series","information security"."testing","BPA","testing"
  Afeed = [Logical_quotient_rating, coding_skills_rating, hackathons, public_speaking_points]

  input_list_col = [self_learning_capability,Extra_courses_did,Taken_inputs_from_seniors_or_elders,worked_in_teams_ever,Introvert,reading_and_writing_skills,memory_capability_score,smart_or_hard_work,Magement_or_Techinical,Interested_subjects,Interested_Type_of_Books,certifications,workshops,Type_of_company_want_to_settle_in,interested_career_area]
  feed = []
  K=0
  j=0
  for i in input_list_col:
    if(i=='Yes'):
      j=2
      feed.append(j)
       
      print("feed 1",i)
    
    elif(i=="No"):
      j=3
      feed.append(j)
       
      print("feed 2",j)
    
    elif(i=='Management'):
      j=1
      k=0
      feed.append(j)
      feed.append(K)
       
      print("feed 10,11",i,j,k)

    elif(i=='Technical'):
      j=0
      k=1
      feed.append(j)
      feed.append(K)
       
      print("feed 12,13",i,j,k)

    elif(i=='Smart worker'):
      j=1
      k=0
      feed.append(j)
      feed.append(K)
       
      print("feed 14,15",i,j,k)

    elif(i=='Hard Worker'):
      j=0
      k=1
      feed.append(j)
      feed.append(K)
      print("feed 16,17",i,j,k)
    
    else:
      for key in Range_dict:
        if(i==key):
          j = Range_dict[key]
          feed.append(j)
         
          print("feed 3",i,j)

      for key in C:
        if(i==key):
          j = C[key]
          feed.append(j)
          
          print("feed 4",i,j)
      
      for key in W:
        if(i==key):
          j = W[key]
          feed.append(j)
          
          print("feed 5",i,j)
      
      for key in ISC:
        if(i==key):
          j = ISC[key]
          feed.append(j)
          
          print("feed 6",i,j)

      for key in ICA:
        if(i==key):
          j = ICA[key]
          feed.append(j)
          
          print("feed 7",i,j)

      for key in TOCO:
        if(i==key):
          j = TOCO[key]
          feed.append(j)
          
          print("feed 8",i,j)

      for key in IB:
        if(i==key):
          j = IB[key]
          feed.append(j)
          
          print("feed 9",i,j)

   
       
  t = Afeed+feed    
  output = regressor.predict([t])
  
  return(output)

def main():
  st.title(" This WebApp takes input from user about various areas and predicts a suitable carrer area :)") #simple title for the app
  html_temp="""
      <div>
      <h2>Career Prediction ML app</h2>
      </div>
      """
  st.markdown(html_temp,unsafe_allow_html=True) #a simple html 
  Name=st.text_input("Full Name")
  Contact_Number=st.text_input("Contact Number")
  Email_address=st.text_input("Email address")
  Logical_quotient_rating = st.selectbox('Rate Your skill(1-0): Logical quotient',('1','2','3','4','5','6','7','8','9','10'))
  coding_skills_rating = st.selectbox('Rate Your skill(1-0):Coding Skills',('1','2','3','4','5','6','7','8','9','10'))
  hackathons = st.selectbox('Rate Your skill(1-0): Hackathons',('1','2','3','4','5','6','7','8','9','10'))
  public_speaking_points = st.selectbox('Rate Your skill(1-0): Public Speaking',('1','2','3','4','5','6','7','8','9','10'))
  self_learning_capability = st.selectbox('Self Learning Capability',('Yes', 'No'))
  Extra_courses_did = st.selectbox('Extra courses',('Yes', 'No'))
  Taken_inputs_from_seniors_or_elders = st.selectbox('Took advice from seniors or elders',('Yes', 'No'))
  worked_in_teams_ever = st.selectbox('Team Co-ordination Skill',('Yes', 'No'))
  Introvert = st.selectbox('Introvert',('Yes', 'No'))
  reading_and_writing_skills = st.selectbox('Reading and writing skills',('poor','medium','excellent'))
  memory_capability_score = st.selectbox('Memory capability score',('poor','medium','excellent'))
  smart_or_hard_work = st.selectbox('Smart or Hard Work',('Smart worker', 'Hard Worker'))
  Magement_or_Techinical = st.selectbox('Management or Techinical',('Management', 'Technical'))
  Interested_subjects = st.selectbox('Interested Subjects',('programming', 'Management', 'data engineering', 'networks', 'Software Engineering', 'cloud computing', 'parallel computing', 'IOT', 'Computer Architecture', 'hacking'))
  Interested_Type_of_Books = st.selectbox('Interested Books Category',('Series', 'Autobiographies', 'Travel', 'Guide', 'Health', 'Journals', 'Anthology', 'Dictionaries', 'Prayer books', 'Art', 'Encyclopedias', 'Religion-Spirituality', 'Action and Adventure', 'Comics', 'Horror', 'Satire', 'Self help', 'History', 'Cookbooks', 'Math', 'Biographies', 'Drama', 'Diaries', 'Science fiction', 'Poetry', 'Romance', 'Science', 'Trilogy', 'Fantasy', 'Childrens', 'Mystery'))
  certifications = st.selectbox('Interested_Type_of_Books',('information security', 'shell programming', 'r programming', 'distro making', 'machine learning', 'full stack', 'hadoop', 'app development', 'python'))
  workshops = st.selectbox('Workshops',('Testing', 'database security', 'game development', 'data science', 'system designing', 'hacking', 'cloud computing', 'web technologies'))
  Type_of_company_want_to_settle_in = st.selectbox('Type of Company You Want to Settle In ',('BPA', 'Cloud Services', 'product development', 'Testing and Maintainance Services', 'SAaS services', 'Web Services', 'Finance', 'Sales and Marketing', 'Product based', 'Service Based'))
  interested_career_area = st.selectbox('Interested Career Area',('testing', 'system developer', 'Business process analyst', 'security', 'developer', 'cloud computing'))
  result=""
  if st.button("Predict"):
    result=inputlist(Name,Contact_Number,Email_address,Logical_quotient_rating, coding_skills_rating, hackathons, public_speaking_points, self_learning_capability,Extra_courses_did, 
                     Taken_inputs_from_seniors_or_elders,worked_in_teams_ever, Introvert,
                     reading_and_writing_skills,memory_capability_score, smart_or_hard_work, 
                     Magement_or_Techinical,Interested_subjects, Interested_Type_of_Books,certifications, 
                     workshops, Type_of_company_want_to_settle_in, interested_career_area) 
    #result will be displayed if button is pressed
    st.success("Predicted Career Option : "
               "{}".format(result))
if __name__=='__main__':
    main()


