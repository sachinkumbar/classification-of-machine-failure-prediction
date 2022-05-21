#pip install streamlit
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

st.title("Machine Failure Prediction")
st.sidebar.header('User inputs :')


#here, 
#1 sands for low 
#2 stands for medium 
#0 stands for high performance

def main():
    choice = st.sidebar.selectbox("Choose how to predict",('Csv File','Column Inputs'))
    if choice == "Csv File":
        upload =st.file_uploader("upload_csv",type=["csv"])
        st.write(upload)
        if upload is not None:
            file0 = pd.read_csv(upload)
            file1 =file0.copy()
            machine = pd.read_csv(r'C:\Users\Admin\Documents\DATA SCIENCE\ExcelR projects\predictive maintaince\Maintaince.csv')
            ## Label_encoder
            from sklearn.preprocessing import LabelEncoder
            label=LabelEncoder()
            machine['Type']=label.fit_transform(machine['Type'])
            # dropping columns
            machine = machine.drop(columns=['UDI', 'Product ID','Rotational speed [rpm]','TWF','HDF','PWF','OSF','RNF'])
            ## Upsampling the data
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42)
            x_res, y_res = sm.fit_resample(machine.iloc[:,:5], machine['Machine failure'])
            machine = pd.concat([x_res, y_res], axis=1)
            ### Normalizing the Data
            x = machine.values 
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df = pd.DataFrame(x_scaled)
            df.columns=(machine.columns)
            ##Test and Train
            train = df.iloc[:,:5]
            test = df.iloc[:,5]
            x_train,x_test,y_train,y_test = train_test_split(train,test,test_size=0.2) 
            ## model
            rf_model=RandomForestClassifier()
            rf_model.fit(x_train,y_train)
            prediction1 = rf_model.predict(file1)
            file1['prediction']=prediction1
            st.write(file1)
            return file1
    elif choice == "Column Inputs":
        def user_input_feature():
            depType =  st.sidebar.selectbox('Type',('0','1','2'))
            depair = st.sidebar.number_input(" Insert the Air temperature ")
            deppro = st.sidebar.number_input(" Insert the Process temperature ")
            deptorq = st.sidebar.number_input(" Insert the Torque")
            deptool = st.sidebar.number_input(" Insert the Tool Wear ")
            data = {'Type':depType,
                    'Air temperature':depair,
                    'Process temperature':deppro,
                    'Torque ':deptorq,
                    'Tool wear':deptool}
            features = pd.DataFrame(data,index = [0])
            return features 
        uf = user_input_feature()
        st.subheader('Input parameters')
        st.write(uf)
        ## importing the data
        machine = pd.read_csv(r'C:\Users\Admin\Documents\DATA SCIENCE\ExcelR projects\predictive maintaince\Maintenance.csv')
        ## Label_encoder
        from sklearn.preprocessing import LabelEncoder
        label=LabelEncoder()
        machine['Type']=label.fit_transform(machine['Type'])
        # dropping columns
        machine = machine.drop(columns=['UDI', 'Product ID','Rotational speed [rpm]','TWF','HDF','PWF','OSF','RNF'])
        ## Upsampling the data
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        x_res, y_res = sm.fit_resample(machine.iloc[:,:5], machine['Machine failure'])
        machine = pd.concat([x_res, y_res], axis=1)
        ### Normalizing the Data
        x = machine.values 
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)
        df.columns=(machine.columns)
        ##Test and Train
        train = df.iloc[:,:5]
        test = df.iloc[:,5]
        x_train,x_test,y_train,y_test = train_test_split(train,test,test_size=0.2) 
        ## model
        rf_model=RandomForestClassifier()
        rf_model.fit(x_train,y_train)
        prediction=rf_model.predict(uf)
        prediction_proba = rf_model.predict_proba(uf)
        st.subheader('Predicted results')
        st.write('Yes it will fail' if prediction == 1  else 'No it will not fail')
        st.subheader('Prediction Probability')
        a = st.write(prediction_proba)
        return a

if __name__=='__main__':
    main()
    
    