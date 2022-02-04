import os
with open(os.path.join('Career-path-Prediction','Procfile'), "w") as file1:
    toFile = 'web: sh setup.sh && streamlit run <app name>.py'
    
file1.write(toFile)
