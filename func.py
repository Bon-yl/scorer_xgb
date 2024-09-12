from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce
import numpy as np
import pickle

def get_train_test(df):
    train_oid, test_oid = train_test_split(df['you_oid'].unique(), test_size=0.1, random_state=42)
    train = df[df['you_oid'].isin(train_oid)]

    test = df[df['you_oid'].isin(test_oid)]
    test = test.dropna(subset=['you_oid','competitor_id','you_breadcrumbs2','competitor_breadcrumbs2','you_host','competitor_host','color_score','openai_score','image_score','mobilenet_score','Exact Match'])
    return train, test
            
# Function to split on the occurrence of '>'
def keep_last_breadcrumbs_levels(df, level):
    def process_breadcrumbs(text, level):
        parts = text.split(' > ')
        if len(parts) > level:
            new_breadcrumbs = ' > '.join(parts[-level:])
        else:
            new_breadcrumbs = text
        return new_breadcrumbs
    df['you_breadcrumbs2'] = df['you_breadcrumbs2'].apply(lambda text : process_breadcrumbs(text, level=level))
    df['competitor_breadcrumbs2'] = df['competitor_breadcrumbs2'].apply(lambda text : process_breadcrumbs(text, level=level))
    return df

def preprocessing(df):        
        columns_to_lower = ['you_breadcrumbs2','competitor_breadcrumbs2','you_host','competitor_host']
        for col in columns_to_lower:
                df[col] = df[col].str.lower()

        df['you_host'] = df['you_host'].apply(lambda x: x.split('.')[1] if x.startswith('www.') else x.split('.')[0])
        df['competitor_host'] = df['competitor_host'].apply(lambda x: x.split('.')[1] if x.startswith('www.') else x.split('.')[0])

        #df['Exact Match'] = df['Exact Match'].apply(lambda x : 0 if x == False else 1)
        df['Exact Match'] = df['Exact Match'].astype(int)
        df.dropna(inplace=True)

        X = df.drop(['Exact Match'], axis=1)
        y = df['Exact Match']

        print(X.shape, y.shape)

        return X , y

def cat_encoding(X, y=None, fit=True, encoder=None):
    cols_to_train = ['you_breadcrumbs2','competitor_breadcrumbs2','you_host','competitor_host','color_score','openai_score','image_score','mobilenet_score']
    if fit:
        encoder = ce.TargetEncoder(cols=['you_breadcrumbs2','competitor_breadcrumbs2','you_host','competitor_host'])
        X_encoded = encoder.fit_transform(X[cols_to_train], y)
    else:
        X_encoded = encoder.transform(X[cols_to_train])
    return X_encoded, encoder

def scaling(df, cols_to_scale, fit=True, scaler=None):
    if fit:
        scaler = MinMaxScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    else:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df, scaler

def remove_outliers(df):
    for col in ["color_score", "openai_score", "image_score", "mobilenet_score"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col][~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
    return df

def evaluation(y_prob, y_test, X_test):
    '''for i in [0.5,0.7,0.9,0.95]:
        print(i)
        y_pred = np.where(y_prob > i, 1, 0)

        # Generate a classification report
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print()
        print(cm[0,1]/(cm[0,1]+cm[1,1])*100)
        print()
        print(cm[1,1]/(cm[1,0]+cm[1,1])*100)
        print()'''

    infer = X_test.copy()

    infer['y_prob'] = y_prob
    infer['y_test'] = y_test
    infer['y_pred'] = np.where(y_prob > 0.5, 1, 0)
    infer['rank'] = infer.groupby('you_oid')['y_prob'].rank(ascending=False, method='first')
    infer['rank'] = infer['rank'].astype(int)
    average_rank = infer[infer['y_test']==1]['rank'].mean()

    y_pred = np.where(y_prob > 0.5, 1, 0)
    stats = classification_report(y_test, y_pred, output_dict = True)['weighted avg']
    cm = confusion_matrix(y_test, y_pred)
    stats['TP'] = cm[1, 1]
    stats['FP'] = cm[0, 1]
    stats['TN'] = cm[0, 0]
    stats['FN'] = cm[1, 0]
    stats['Avg score (positive class)'] = infer[infer['y_pred']==1]['y_prob'].mean()
    stats['Avg score (negative class)'] = infer[infer['y_pred']==0]['y_prob'].mean()
    stats['Average rank'] = average_rank
    
    #print(stats)
    return infer, stats

# save encoder and model
def save_model_encoder(model, encoder, model_name, encoder_name):
    with open(encoder_name, 'wb') as f:
        pickle.dump(encoder, f)

    with open(model_name, 'wb') as f:
        pickle.dump(model, f)