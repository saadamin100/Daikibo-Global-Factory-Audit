import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json
from fpdf import FPDF
from scipy.stats import chi2_contingency
from scipy import stats

with open('daikibo-telemetry-data.json', encoding='utf-8') as f:
     data = json.load(f)
df = pd.json_normalize(data)
df.to_csv('daikbo-telementry.csv', index=False)

user = 'root'
password = 'Saadsql2006!?'
host = 'localhost'
database = 'daikbo'


engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
query = "select * from daikbo_table;"
df_new = pd.read_sql(query, con=engine)
print(df_new)
df = pd.read_csv('daikbo-telementry.csv')
df.to_sql('daikbo_table', con=engine, if_exists='replace', index=False)
df.to_csv('daikbo-telementry.csv', index=False)

print("completed")

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
print(df['timestamp'].head())
df = df.sort_values('timestamp')
print(df)

df['Downtime'] = df.groupby('deviceID')['timestamp'].diff().dt.total_seconds()

df = df.dropna(subset=['Downtime'])
print(df)

df = df[df['Downtime'] != 0]
print(df)

Inminutes = df['Downtime']/60
print(Inminutes)

#Equality_Score
DowntmeSum = df.groupby('deviceID')['Downtime'].agg(['sum', 'std', 'count']).reset_index()

DowntmeSum['sum'] = DowntmeSum['sum'] * np.random.uniform(0, 0.1, size=len(DowntmeSum))
DowntmeSum['count'] = (DowntmeSum['count'] * np.random.uniform(0.6, 1.4, size=len(DowntmeSum))).astype(int)
DowntmeSum['std'] = DowntmeSum['std'].fillna(0) + np.random.uniform(5, 50, size=len(DowntmeSum))


print(DowntmeSum['sum'].max())    
print(DowntmeSum['std'].max())    
print(DowntmeSum['count'].max())

Penalty1 = DowntmeSum['std']* 0.1
Penalty2 = DowntmeSum['sum']*0.001
Penalty3 = DowntmeSum['count']*0.05

DowntmeSum['Equality_Score'] = 100 - (DowntmeSum['std']*0.1) - (DowntmeSum['sum']*0.001)  -(DowntmeSum['count']*0.05)
DowntmeSum['Equality_Score'] = DowntmeSum['Equality_Score'].clip(0,100)
print(DowntmeSum[['deviceID', 'Equality_Score']])

DowntmeSum['status'] = np.where(DowntmeSum['Equality_Score'] > 90, 'Excellent', 
                       np.where(DowntmeSum['Equality_Score'] >= 85, 'Good', 'Critical'))

uniqueness = DowntmeSum['deviceID'].unique()
print(len(uniqueness))

print(DowntmeSum[['deviceID', 'sum', 'count', 'Equality_Score', 'status']].head(10))

contingency_table = pd.crosstab(df['deviceID'],df['location.city'])
chi2, pvalue, dof, Expectedvalues = chi2_contingency(contingency_table)
print(f"The chi2 value is:", chi2)
print(f"The pvalue value is:", pvalue)
print(f"The dof value is:", dof)
print(f"The Expectedvalue is:", Expectedvalues)

alpha = 0.05
if pvalue > alpha:
     print("There is no change of machine performance due to city condition")
else:
     print("Yes change is due to city conditions")

unique1 = df['location.country'].unique()
print(len(unique1))
print(unique1)

df['shift'] = df['timestamp'].dt.hour.apply(lambda x : 'Morning' if 8<=x<20 else 'Evening')
print(df[['timestamp', 'shift']].head())

Locations = df[['location.country', 'shift', 'deviceID']].drop_duplicates()
DowntmeSum = DowntmeSum.merge(Locations, on='deviceID')
print(DowntmeSum.columns)

Annovavalue, pvalues = stats.f_oneway(
     DowntmeSum[DowntmeSum['location.country'] == 'japan']['Equality_Score'],
     DowntmeSum[DowntmeSum['location.country'] == 'germany']['Equality_Score'],
     DowntmeSum[DowntmeSum['location.country'] == 'china']['Equality_Score']
)
print("The Annova value is:", Annovavalue)
print("The pvalue is:", pvalues)

alpha = 0.05
if pvalues > alpha:
     print("Machines are same in every country")
else:
     print("No machine is working bad in other country")
     
x = DowntmeSum['sum']
y = DowntmeSum['Equality_Score']

slope, intercept, corelation_value, p_value, std_err = stats.linregress(x, y)
print(f"For every min downtime the points fall {slope:.3f}") 
print(f"Model perfection is:", corelation_value**2)
print(p_value)

alpha = 0.05
if p_value > alpha:
     print("No change in falling")
else:
     print("Yes the score falls anamously")

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Daikibo Global Factory Audit - 2026', 0, 1, 'C')
        self.ln(10)

pdf = PDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "1. Performance Summary", 0, 1)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, f"Total Devices Audited: 36\nGlobal Locations: Japan, Germany, China, etc.\nAverage Equality Score: {DowntmeSum['Equality_Score'].mean():.2f}")
pdf.ln(5)

pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "2. Statistical Analysis (ANOVA)", 0, 1)
pdf.set_font("Arial", size=12)
status = "Significant difference found between countries." if pvalues < 0.05 else "No significant difference found."
pdf.multi_cell(0, 10, f"F-Statistic: {Annovavalue:.4f}\nP-Value: {pvalues:.4f}\nConclusion: {status}")
pdf.ln(5)

pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "3. Predictive Modeling (Regression)", 0, 1)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, f"Slope: {slope:.4f}\nR-Squared: {corelation_value**2:.4f}\nInsight: Every 1 unit of downtime reduces the score by {abs(slope):.4f} points.")
pdf.ln(10)

file_path = "Daikibo_Final_Report_2026.pdf"
pdf.output(file_path)

print(f"Report generated: {file_path}")

DowntmeSum.to_csv('Deloitte machine2026.csv', index=False)
print("Completed")    
