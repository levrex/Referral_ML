import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import importlib as imp
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb


def bookkeepingMedication(df_pat, first_date, tol):
    """
    Functions: 
     - Ligate subsequent prescriptions of drugs
        - And ligate medication in a co-therapy setting
     - Fill in the gaps in treatment trajectory to infer effectiveness of drug 
         - This is done with respect to the user specified tolerance (tol)
         - We presume that the prescription is continued (e.g. by GP) or the patient achieved 
             drug-free remission, either way we count it as a success!
        - We call these gap-filling drugs -> ghost drugs
     - Drop drug if it is not prescribed again (even after considering the tolerance)
    
    
    Input:
        df_pat = treatment prescriptions of one patient
        first_date = first prescription date
        tol = tolerance used to fill in the gaps in the prescriptions
    
    Output:
        df_pat_imp = Prescription table where each row represents a unique drug. 
        df_ligated = Prescription table where all overlapping drugs are ligated
        df_ghost = Prescription table with inferred drugs to fill in the gaps on certain occassions
        
    """
    # Fill Nan
    df_pat.loc[:, 'end'] = df_pat['end'].fillna(df_pat['start'])
    df_pat.loc[:, 'start'] = df_pat['start'].fillna(df_pat['end'])
    
    df_pat = df_pat.dropna(subset=['start', 'end'])
    
    df_pat['Ghost'] = False 
    df_pat_imp = pd.DataFrame(columns= df_pat.columns)
    df_ligated = pd.DataFrame(columns = ['ID', 'Drug', 'Time', 'Start', 'End']) # keeps track of longest trajections
    df_ghost = pd.DataFrame(columns = ['ID', 'Drug', 'Time', 'Start', 'End'])
    
    therapy_id = 0 # keep track of unique therapies
    d_cur = {} # keep track of current drug(s)
    l_prev = [] # keep track of previous drug(s) (1 iteration prior)
    l_cur = []
    
    # Fill in the gaps, it they are smaller than tolerance
    for ix, row in df_pat.iterrows():
        row['treatID'] = len(df_pat_imp)
        df_pat_imp.loc[len(df_pat_imp)] = row
        
        end = row['end']
        duration = row['end'] - row['start'] 
        drug = row['Drug']  
        df_future = df_pat[((df_pat['Drug']==drug) & (df_pat['start']> end))].sort_values(by='start').copy()
        if len(df_future) != 0 and end != np.nan  : #if duration < 2:
            if df_future.iloc[0]['start'] - end < tol: # Check if drug is prescribed again within 1 month
                # Fill gap if it is less than 1 month difference
                df_pat.at[ix, 'end'] = df_future.iloc[0]['start']
                
                # create ghost drug
                ghost = row.copy()
                ghost['treatID'] = len(df_pat_imp)
                ghost['Ghost'] = True
                ghost['start'] = row['end']
                
                # End = where next one starts
                ghost['end'] = df_future.iloc[0]['start']
                ghost['daysTreatment'] = ghost['end'] -ghost['start']
                ghost['StartDateTreatment'] = first_date + pd.DateOffset(days=int(ghost['start']))
                ghost['EndDateTreatment'] = first_date + pd.DateOffset(days=int(ghost['end']))
                df_pat_imp.loc[len(df_pat_imp)] = ghost
    
    # Remove all duplicates
    df_pat_imp = df_pat_imp.drop_duplicates(subset=['Drug', 'StartDateTreatment','EndDateTreatment' ])
    
    # Remove all nans
    df_pat_imp = df_pat_imp.dropna(subset=['Drug', 'StartDateTreatment','EndDateTreatment'])
    
    # Get all dates
    l_hist = list(df_pat['start'].unique())
    l_hist.extend(list(df_pat['end'].unique()))
    l_hist = list(set(l_hist))
    l_hist.sort()
    
    # Loop through all dates to check whether the current therapic treatment is updated
    for ix, date in enumerate(l_hist):
        l_cur = []
        sub_df = df_pat[((df_pat['start'] <= date) & (df_pat['end'] > date ))].copy()
        if ix != len(l_hist)-1:
            end_date = min(l_hist[ix+1], sub_df['end'].min()) # Smallest end date -> (army can move just as fast as the slowest soldier)
        else :
            end_date = sub_df['end'].min()
        
        l_cur = list(sub_df['Drug'].unique())
        if set(l_cur) == set(l_prev) and l_prev != []:
            df_ligated.at[therapy_id, 'End'] = end_date # update end date
            df_ligated.at[therapy_id, 'Time'] = end_date - df_ligated.at[therapy_id, 'Start']
        elif set(l_cur) != set(l_prev) and l_cur != []: # dont add empty
            # Add new instance/therapy in the patient trajectory
            therapy_id += 1
            drug_str = ''
            for i, entry in enumerate(sub_df.iterrows()):
                row = entry[1]
                if row['Drug'] not in drug_str:
                    if drug_str == '':
                        drug_str += row['Drug']
                    else :
                        drug_str += ' & ' + row['Drug']
            duration = end_date - date
            df_ligated.loc[therapy_id, :] = [therapy_id, drug_str, duration, date, end_date]
            
        l_prev = l_cur.copy()
    
    # Infer effective therapy
    prev_drug = ''
    ghost_id = 0
    for ix, row in df_ligated.iterrows():
        if row['Drug'] == prev_drug:
            df_ghost.at[ghost_id, 'End'] = row['End'] # update end date
        else : 
            ghost_id += 1
            duration = row['End'] - row['Start']
            df_ghost.loc[ghost_id, :] = [ghost_id, row['Drug'], duration, row['Start'], row['End']]
            prev_drug = row['Drug']
    return df_pat_imp, df_ligated, df_ghost


def getTableEffectiveRX(df_treat, tol=183, ignore_prednison=True, output=False, verbose=False):
    """
    Input:
        df_treat = dataframe with all medication information
        ignore_prednison = boolean indicating whether or not to ignore prednison
        tol = tolerance used to fill in the gaps in the prescriptions
        verbose = whether or not to print out extra information
        output = return underlying treatment information
        
    Output:
        df_effect = Table where effective Rx is collected for each patient
        
    Extra Output (only returned when output=True):
        df_drugs = Prescription table where all overlapping drugs are ligated to reveal trajectories
        df_drugs_inf = Prescription table with inferred drugs to fill in the gaps on specific occassions
    
    Description:
        Generate a table for the effective Rx (a.k.a. the drug that patients stay on)
        This is a proxy for the success of a treatment.
    """
    df_effect = pd.DataFrame(columns = ['pseudoId', 'Drug', 'Time', 'Start', 'End', 'Date', 'EndDate'])
    df_drugs = pd.DataFrame(columns = ['pseudoId', 'Drug', 'Time', 'Start', 'End', 'Date', 'EndDate'])
    df_drugs_inf = pd.DataFrame(columns = ['pseudoId', 'Drug', 'Time', 'Start', 'End', 'Date', 'EndDate'])
    
    for pat in df_treat['pseudoId'].unique():
        if verbose : print(pat)
        df_pat = df_treat[df_treat['pseudoId']==pat].copy()
        if ignore_prednison:
            df_pat = df_pat[~(df_pat['Drug'].isin(['PREDNISOLON', 'METHYLPREDNISOLON']))]

        sub_df = df_pat.copy()

        sub_df = sub_df.dropna(subset=['treatID'])
        
        first_date = pd.to_datetime(pd.Series([i for i in sub_df['StartDateTreatment'] if type(i) != float]).min(), format='%Y-%m-%d', errors='ignore')

        df_pat.loc[:, 'StartDateTreatment'] = pd.to_datetime(df_pat['StartDateTreatment'])
        df_pat.loc[:, 'EndDateTreatment'] = pd.to_datetime(df_pat['EndDateTreatment'])
        #df_pat.loc[:, 'dateDAS'] = pd.to_datetime(df_pat['dateDAS'])

        df_pat = df_pat.sort_values(by=['StartDateTreatment'])
        df_pat['daysTreatment'] = df_pat['EndDateTreatment'] -df_pat['StartDateTreatment']
        df_pat['start'] = df_pat['StartDateTreatment'] - first_date
        df_pat['daysTreatment'] = df_pat['daysTreatment'].dt.days
        df_pat['start'] = df_pat['start'].dt.days
        df_pat['end'] = df_pat['EndDateTreatment']- first_date
        df_pat['end'] = df_pat['end'].dt.days
        

        df_pat, df_ligated, df_ghost = bookkeepingMedication(df_pat, first_date, tol=tol)
        df_ghost = df_ghost.dropna(subset=['Start', 'End'])
        df_ligated = df_ligated.dropna(subset=['Start', 'End'])
        
        if output:
            if len(df_ligated) > 0 and len(df_ghost) > 0:
                # Add actual dates ligated
                df_ligated['pseudoId'] = pat
                df_ligated['Date'] = df_ligated['Start'].apply(lambda x: first_date + pd.DateOffset(int(x)))
                df_ligated['EndDate'] = df_ligated['End'].apply(lambda x: first_date + pd.DateOffset(int(x)))
                # Add actual dates ghost drugs
                df_ghost['pseudoId'] = pat
                
                try:
                    df_ghost['Date'] =  df_ghost['Start'].apply(lambda x: first_date + pd.DateOffset(int(x)))
                except : 
                    print(df_ghost['Start'])
                    print(df_ghost['Start'].unique())
                    print(eql)
                df_ghost['EndDate'] = df_ghost['End'].apply(lambda x: first_date + pd.DateOffset(int(x)))
                # Create table for all patients
                df_drugs = df_drugs.append(df_ligated[['pseudoId', 'Drug', 'Time', 'Start', 'End', 'Date', 'EndDate']])
                df_drugs_inf = df_drugs_inf.append(df_ghost[['pseudoId', 'Drug', 'Time', 'Start', 'End', 'Date', 'EndDate']])
            else :
                print('No drugs found for pt', pat)
                df_drugs = df_drugs.append([pat, '', 0, 0, 0, '', ''])
                df_drugs_inf = df_drugs_inf.append([pat, '', 0, 0, 0, '', ''])
        
        df_ghost['Time'] = df_ghost['Time'].astype(int)
        ix = len(df_effect)
        if len(df_ghost) > 0:
            df_effect.loc[ix] = df_ghost.loc[df_ghost['Time'].idxmax()]#.copy()
        else :
            print('No drugs found for pt', pat)
        df_effect.at[ix, 'pseudoId'] = pat
    if output:
        return df_effect, df_drugs, df_drugs_inf # be sure to add the dates
    else : 
        return df_effect
    

def getTrajectoryRX(df_pat, tol=183, ignore_prednison=True, figure=1):
    """
    Input:
        df_pat = dataframe with all treatment entries of a specific patient
        tol = tolerance used to fill in the gaps in the prescriptions
        ignore_prednison = whether or not to ignore glucocorticosteroids like prednison
        output = whether or not to return the underlying prescription information that informs the figure
        
    Output:
        ax = treatment trajectory figure of 1 patient
        
    Extra Output (only returned when output=True):
        df_ligated = Prescription table where all overlapping drugs are ligated
        df_ghost = Prescription table with inferred drugs to fill in the gaps on specific occassions
    
    Description:
        Visualize the medication trajectory and DAS-score over time for a specific patient
    """
    l_alpha = []
    d_color ={'TOCILIZUMAB' : 'm', 'METHOTREXAAT': 'r', 'HYDROXYCHLOROQUINE': 'b',
       'AZATHIOPRINE': 'orange', 'INFLIXIMAB': 'silver', 'ABATACEPT': 'tan', 
       'SULFASALAZINE': 'g', 'ETANERCEPT': 'lime', 'LEFLUNOMIDE': 'c', 'CERTOLIZUMAB PEGOL': 'purple',
       'CICLOSPORINE': 'darkviolet', 'ADALIMUMAB': 'olive', 'BARICITINIB': 'peachpuff', 
       'GOLIMUMAB': 'hotpink', 'AUROTHIOBARNSTEENZUUR (DI-NA-ZOUT)': 'palegreen',
       'CYCLOFOSFAMIDE': 'coral', 'RITUXIMAB': 'darkred', 'TOFACITINIB': 'chocolate', 'No medication':'gray',
             'SARILUMAB' : 'cyan', 'FILGOTINIB' : 'teal', 'UPADACITINIB' : 'y', 
             }
    # 'PREDNISOLON': 'y', 'METHYLPREDNISOLON': 'teal',
    
    if ignore_prednison:
        df_pat = df_pat[(df_pat['Drug'].isin(d_color))]
        #df_pat = df_pat[~(df_pat['Drug'].isin(['PREDNISOLON', 'METHYLPREDNISOLON', 'FOLIUMZUUR','IBUPROFEN', 'OMEPRAZOL', 'RISEDRONINEZUUR/CALCIUMCARB']))]
    
    sub_df = df_pat.copy()
    sub_df = sub_df.dropna(subset=['treatID'])
    first_date = pd.to_datetime(pd.Series([i for i in sub_df['StartDateTreatment'] if type(i) != float]).min(), format='%Y-%m-%d', errors='ignore') #- pd.DateOffset(days=sub_df['daysTreatment'].min())
    
    
    pd.set_option('chained_assignment',None) # Turn off chained assignment warnings because they are unnecessary in this case
    
    df_pat.loc[:, 'StartDateTreatment'] = pd.to_datetime(df_pat['StartDateTreatment'])
    df_pat.loc[:, 'EndDateTreatment'] = pd.to_datetime(df_pat['EndDateTreatment'])
    #df_pat.loc[:, 'dateDAS'] = pd.to_datetime(df_pat['dateDAS'])
    
    # Calculate start and duration of drugs back to number of days
    df_pat = df_pat.sort_values(by=['StartDateTreatment'])#.copy()
    df_pat.loc[:, 'daysTreatment'] = df_pat['EndDateTreatment'] -df_pat['StartDateTreatment']
    df_pat.loc[:, 'start'] = df_pat['StartDateTreatment'] - first_date
    df_pat.loc[:, 'daysTreatment'] = df_pat['daysTreatment'].dt.days
    df_pat.loc[:, 'start'] = df_pat['start'].dt.days
    df_pat.loc[:, 'end'] = df_pat['EndDateTreatment']- first_date
    df_pat.loc[:, 'end'] = df_pat['end'].dt.days
    
    
    df_pat, df_ligated, df_ghost = bookkeepingMedication(df_pat, first_date, tol=tol)
    
    sub_df = df_pat.copy()
    sub_df = sub_df.drop_duplicates(subset=['treatID'])
    sub_df = sub_df.dropna(subset=['treatID'])
    width = np.array(sub_df['daysTreatment'])
    start  = np.array(sub_df['start'].values) 
    y      = range(len(sub_df)) 
    
    l_col = []
    
    # Make inferred ghost drugs transparent
    for ix, entry in sub_df.iterrows():
        r, g, b = to_rgb(d_color[entry['Drug']])
        if entry['Ghost']:
            alpha = 0.25
        else :
            alpha = 1
        l_col.append([r, g, b, alpha])

    # Generate Legend automatically (select relevant medication)
    d_sub = {}
    for val in np.unique(sub_df['Drug']):
        d_sub[val] = d_color[val]
    
    if figure != 0:
        ax = plt.subplot(1, 1, 1)
        ax.barh(y, width=width, left=start, color=l_col)
        markers = [pl.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in d_sub.values()]
        ax.legend(markers, d_sub.keys(), numpoints=1, bbox_to_anchor=(1.05, 1), loc='upper left',)
    
    if figure==2:
        return ax, df_ligated, df_ghost
    elif figure==1 :
        # By default: only return figure
        return ax
    if figure==0 :
        return df_ligated, df_ghost
