#!/usr/bin/env python
# coding: utf-8

# ### Problem Description

# The data science team in your company was working on a machine learning model that can help doctors in diagnosing diabetes. Then, the deployment team decided that the model itself (given in `model.h5` file) will be on server side so you have to provide the following:

# In[ ]:





# - A scoring script that uses the h5 file to predict the outcome of each patient.
# - The given file `pima-indians-diabetes.data.csv` should be injected to the database under the name of **diabetes_unscored**
# - Your script must listen to the database and take the newly added records in **diabetes_unscored**, run the model on them, and put them back in a new table **diabetes_scored**.
# - Your script should be a scheduled task that will run every hour.

# So, the deployment team will be able to inject data in a table and retrieve the prediction output from the other table.

# #### Best of luck!

# In[147]:


# Let's import sqlalchemy 
import sqlalchemy as db
# create connection with the database
con = db.create_engine('postgresql://iti:iti@localhost/hisham')
# Find out the tables in this DB
con.table_names()


# In[162]:


import pandas as pd
# Create a SQL query to load the entire diabetes table
query = """
select Pregnancies , Glucose ,
BloodPressure ,
SkinThickness ,
Insulin ,
BMI ,
DiabetesPedigreeFunction ,
Age from hisham.public."diabetes_unscored" 
Except
select Pregnancies , Glucose ,
BloodPressure ,
SkinThickness ,
Insulin ,
BMI ,
DiabetesPedigreeFunction ,
Age from hisham.public."diabetes_scored" ;

"""
# Load the table 
diabetes = pd.read_sql(query, con)
# View the head

diabetes.head(60)


# In[163]:


import numpy as np
import json
from keras.models import model_from_json
json_file = open('/home/hihsma/Downloads/Task 3/model.json', 'r')
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights("/home/hihsma/Downloads/Task 3/model.h5")


# In[164]:


arr = diabetes.to_numpy()


# In[165]:


prediction = loaded_model.predict(arr)
print(prediction)


# In[166]:


new_list=[]
for x in prediction :
        for y in x:
            if y >= .5 :
                y=1
            else :
                y=0
            new_list.append(y)


# In[167]:




# In[168]:


diabetes['outcome']= new_list


# In[169]:





# In[170]:


diabetes.to_sql(name = 'diabetes_scored',                           # New table name
                con=con,                                            # Connection object to the database
                schema = 'public',index = False ,                                  # Name of schema to store the data in
                if_exists='append')                                 # Action to be done if a table with the same name exists


# In[ ]:



print('done')

# In[ ]:





# In[ ]:




