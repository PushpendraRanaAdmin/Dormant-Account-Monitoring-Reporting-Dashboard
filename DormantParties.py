import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Load the data
df = pd.read_excel("C:/Users/6th/Desktop/Backup/Surv Sales Coordinator/Dormant Accs/Analyze Dormants.xlsx", sheet_name='Sheet1')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Number of unique accounts: {df['Account'].nunique()}")
print(f"Date range: {df['Month'].min()} to {df['Month'].max()}")

# Convert Month to datetime
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')

# Get current date for analysis
current_date = datetime.now()

# 1. ANALYSIS 1: Account Purchase Frequency Analysis
print("\n" + "=" * 50)
print("ANALYSIS 1: ACCOUNT PURCHASE FREQUENCY")
print("=" * 50)

# Calculate purchase frequency per account
account_purchase_stats = df.groupby('Account').agg({
    'Month': ['count', 'max', 'min'],
    'SalesAmount': 'sum',
    'Quantity': 'sum',
    'Item Name': 'nunique'
}).round(2)

# Flatten column names
account_purchase_stats.columns = ['Purchase_Count', 'Last_Purchase_Date', 'First_Purchase_Date',
                                  'Total_Revenue', 'Total_Quantity', 'Unique_Products']

# Calculate days since last purchase
account_purchase_stats['Days_Since_Last_Purchase'] = (
            current_date - account_purchase_stats['Last_Purchase_Date']).dt.days


# Calculate average days between purchases (for accounts with >1 purchase)
def calculate_avg_days_between(group):
    if len(group) > 1:
        dates = sorted(group['Month'].unique())
        gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        return np.mean(gaps) if gaps else np.nan
    else:
        return np.nan


avg_days_df = df.groupby('Account').apply(calculate_avg_days_between)
account_purchase_stats['Avg_Days_Between_Purchases'] = avg_days_df


# 2. Categorize accounts
def categorize_account(row):
    if row['Purchase_Count'] == 1:
        return 'One-Time Buyer'
    elif row['Purchase_Count'] == 2:
        return 'Two-Time Buyer'
    elif row['Purchase_Count'] == 3:
        return 'Three-Time Buyer'
    elif row['Purchase_Count'] <= 5:
        return 'Low-Frequency (4-5 purchases)'
    elif row['Days_Since_Last_Purchase'] > 180:
        return 'Dormant (>6 months)'
    elif row['Days_Since_Last_Purchase'] > 90:
        return 'At Risk (3-6 months)'
    elif row['Days_Since_Last_Purchase'] > 30:
        return 'Recent (1-3 months)'
    else:
        return 'Active (<1 month)'


account_purchase_stats['Account_Category'] = account_purchase_stats.apply(categorize_account, axis=1)

# Add location info
account_info = df.groupby('Account').agg({
    'City': 'first',
    'State': 'first',
    'Branch': 'first',
    'Salesman': lambda x: ', '.join(set(x))
}).rename(columns={'Salesman': 'Assigned_Salesmen'})

account_analysis = pd.concat([account_purchase_stats, account_info], axis=1)

# 3. ANALYSIS 2: One-Time Buyers Analysis
print("\n" + "=" * 50)
print("ANALYSIS 2: ONE-TIME BUYERS")
print("=" * 50)

# Reset index to make Account a column
account_analysis_reset = account_analysis.reset_index()

one_time_buyers = account_analysis_reset[account_analysis_reset['Purchase_Count'] == 1]
print(f"Number of one-time buyers: {len(one_time_buyers)}")
print(f"Percentage of all accounts: {len(one_time_buyers) / len(account_analysis_reset) * 100:.1f}%")
print(f"Total revenue from one-time buyers: ₹{one_time_buyers['Total_Revenue'].sum():,.0f}")

# Top one-time buyers by revenue
top_one_time = one_time_buyers.sort_values('Total_Revenue', ascending=False).head(10)
print("\nTop 10 One-Time Buyers by Revenue:")
print(top_one_time[['Account', 'City', 'State', 'Total_Revenue', 'Last_Purchase_Date']].to_string(index=False))

# 4. ANALYSIS 3: Low-Frequency Buyers (2-5 purchases)
print("\n" + "=" * 50)
print("ANALYSIS 3: LOW-FREQUENCY BUYERS (2-5 purchases)")
print("=" * 50)

low_freq_buyers = account_analysis_reset[account_analysis_reset['Purchase_Count'].between(2, 5)]
print(f"Number of low-frequency buyers: {len(low_freq_buyers)}")
print(f"Percentage of all accounts: {len(low_freq_buyers) / len(account_analysis_reset) * 100:.1f}%")

# 5. ANALYSIS 4: Dormant Accounts (>90 days)
print("\n" + "=" * 50)
print("ANALYSIS 4: DORMANT & AT-RISK ACCOUNTS")
print("=" * 50)

dormant_accounts = account_analysis_reset[account_analysis_reset['Days_Since_Last_Purchase'] > 90]
print(f"Number of dormant accounts (>90 days): {len(dormant_accounts)}")

# Categorize by dormancy period
dormant_accounts['Dormancy_Period'] = pd.cut(dormant_accounts['Days_Since_Last_Purchase'],
                                             bins=[90, 180, 270, 365, float('inf')],
                                             labels=['3-6 months', '6-9 months', '9-12 months', '>12 months'])

dormant_summary = dormant_accounts.groupby('Dormancy_Period').agg({
    'Account': 'count',
    'Total_Revenue': 'sum'
}).rename(columns={'Account': 'Count'})

print("\nDormant Accounts by Period:")
print(dormant_summary.to_string())

# 6. ANALYSIS 5: High-Value Dormant Accounts
print("\n" + "=" * 50)
print("ANALYSIS 5: HIGH-VALUE DORMANT ACCOUNTS")
print("=" * 50)

# Define high-value threshold (top 20% by revenue)
revenue_threshold = account_analysis_reset['Total_Revenue'].quantile(0.8)
high_value_dormant = dormant_accounts[dormant_accounts['Total_Revenue'] > revenue_threshold]

print(f"High-value dormant accounts (revenue > ₹{revenue_threshold:,.0f}): {len(high_value_dormant)}")

print("\nTop 10 High-Value Dormant Accounts:")
top_dormant = high_value_dormant.sort_values('Total_Revenue', ascending=False).head(10)
print(top_dormant[['Account', 'City', 'State', 'Total_Revenue', 'Days_Since_Last_Purchase']].to_string(index=False))

# 7. Create comprehensive report
report_data = []

for _, row in account_analysis_reset.iterrows():
    report_data.append({
        'Account': row['Account'],
        'City': row['City'],
        'State': row['State'],
        'Branch': row['Branch'],
        'Purchase_Count': row['Purchase_Count'],
        'Total_Revenue': row['Total_Revenue'],
        'Last_Purchase_Date': row['Last_Purchase_Date'].strftime('%Y-%m-%d'),
        'Days_Since_Last_Purchase': row['Days_Since_Last_Purchase'],
        'Account_Category': row['Account_Category'],
        'Avg_Days_Between_Purchases': row['Avg_Days_Between_Purchases'],
        'Unique_Products': row['Unique_Products'],
        'Assigned_Salesmen': row['Assigned_Salesmen'],
        'Priority_Score': (row['Total_Revenue'] / 10000) * (100 / row['Days_Since_Last_Purchase']) if row[
                                                                                                          'Days_Since_Last_Purchase'] > 0 else 0
    })

report_df = pd.DataFrame(report_data)

# Add priority flag
report_df['Action_Priority'] = pd.cut(report_df['Priority_Score'],
                                      bins=[-1, 1, 5, 10, float('inf')],
                                      labels=['Low', 'Medium', 'High', 'Critical'])

# 8. ANALYSIS 6: Salesman Performance by Account Health
print("\n" + "=" * 50)
print("ANALYSIS 6: SALESMAN PERFORMANCE")
print("=" * 50)

# Create salesman analysis
salesman_analysis = []
for salesman in df['Salesman'].unique():
    salesman_accounts = df[df['Salesman'] == salesman]['Account'].unique()
    salesman_df = report_df[report_df['Account'].isin(salesman_accounts)]

    if len(salesman_df) > 0:
        salesman_analysis.append({
            'Salesman': salesman,
            'Total_Accounts': len(salesman_df),
            'Dormant_Accounts': len(salesman_df[salesman_df['Days_Since_Last_Purchase'] > 90]),
            'One_Time_Buyers': len(salesman_df[salesman_df['Purchase_Count'] == 1]),
            'Total_Revenue': salesman_df['Total_Revenue'].sum(),
            'Avg_Account_Revenue': salesman_df['Total_Revenue'].mean()
        })

salesman_df = pd.DataFrame(salesman_analysis)

print("\nSalesman with Most Dormant Accounts:")
print(salesman_df.sort_values('Dormant_Accounts', ascending=False).head(10)[
          ['Salesman', 'Dormant_Accounts', 'Total_Accounts']].to_string(index=False))

# 9. ANALYSIS 7: Product-wise Analysis for Dormant Accounts
print("\n" + "=" * 50)
print("ANALYSIS 7: PRODUCT ANALYSIS FOR DORMANT ACCOUNTS")
print("=" * 50)

# Get products purchased by dormant accounts
dormant_account_list = report_df[report_df['Days_Since_Last_Purchase'] > 90]['Account'].tolist()
dormant_products = df[df['Account'].isin(dormant_account_list)].groupby('Sub Group').agg({
    'SalesAmount': 'sum',
    'Quantity': 'sum',
    'Account': 'nunique'
}).rename(columns={'Account': 'Number_of_Dormant_Accounts'})

print("\nTop Products Purchased by Dormant Accounts:")
print(dormant_products.sort_values('SalesAmount', ascending=False).head(10).to_string())

# 10. Save reports to Excel
with pd.ExcelWriter('Account_Analysis_Reports.xlsx') as writer:
    # Summary sheet
    summary_stats = pd.DataFrame({
        'Metric': ['Total Accounts', 'One-Time Buyers', 'Low-Frequency Buyers',
                   'Dormant Accounts (>90 days)', 'High-Value Dormant Accounts',
                   'Active Accounts', 'Total Revenue'],
        'Value': [len(account_analysis_reset), len(one_time_buyers), len(low_freq_buyers),
                  len(dormant_accounts), len(high_value_dormant),
                  len(account_analysis_reset) - len(dormant_accounts),
                  f"₹{account_analysis_reset['Total_Revenue'].sum():,.0f}"]
    })
    summary_stats.to_excel(writer, sheet_name='Summary', index=False)

    # Detailed account analysis
    report_df_sorted = report_df.sort_values(['Action_Priority', 'Priority_Score'], ascending=[False, False])
    report_df_sorted.to_excel(writer, sheet_name='Account_Analysis', index=False)

    # One-time buyers
    one_time_buyers_sorted = report_df[report_df['Purchase_Count'] == 1].sort_values('Total_Revenue', ascending=False)
    one_time_buyers_sorted.to_excel(writer, sheet_name='One_Time_Buyers', index=False)

    # Dormant accounts
    dormant_sorted = report_df[report_df['Days_Since_Last_Purchase'] > 90].sort_values(
        ['Total_Revenue', 'Days_Since_Last_Purchase'], ascending=[False, False])
    dormant_sorted.to_excel(writer, sheet_name='Dormant_Accounts', index=False)

    # High-priority accounts
    high_priority = report_df[report_df['Action_Priority'].isin(['High', 'Critical'])].sort_values(
        'Priority_Score', ascending=False)
    high_priority.to_excel(writer, sheet_name='High_Priority', index=False)

    # Salesman performance
    salesman_df.to_excel(writer, sheet_name='Salesman_Performance', index=False)

    # Product analysis
    dormant_products.to_excel(writer, sheet_name='Dormant_Products')

print("\n" + "=" * 50)
print("REPORTS GENERATED")
print("=" * 50)
print(f"Detailed reports saved to 'Account_Analysis_Reports.xlsx'")
print("\nSheets included:")
print("1. Summary - Key metrics overview")
print("2. Account_Analysis - All accounts with analysis")
print("3. One_Time_Buyers - Accounts with only 1 purchase")
print("4. Dormant_Accounts - Accounts inactive >90 days")
print("5. High_Priority - Accounts needing immediate attention")
print("6. Salesman_Performance - Performance by salesman")
print("7. Dormant_Products - Product analysis for dormant accounts")

# 11. Visualizations
plt.figure(figsize=(16, 12))

# Plot 1: Account categories
plt.subplot(3, 3, 1)
category_counts = report_df['Account_Category'].value_counts()
colors = plt.cm.Set3(np.arange(len(category_counts)))
category_counts.plot(kind='bar', color=colors)
plt.title('Account Categories Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Category')
plt.ylabel('Number of Accounts')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Plot 2: Purchase frequency distribution
plt.subplot(3, 3, 2)
freq_dist = report_df['Purchase_Count'].value_counts().sort_index().head(15)
freq_dist.plot(kind='bar', color='lightcoral')
plt.title('Purchase Frequency Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Number of Purchases')
plt.ylabel('Number of Accounts')
plt.axhline(y=1, color='red', linestyle='--', alpha=0.3)

# Plot 3: Revenue by account category
plt.subplot(3, 3, 3)
revenue_by_category = report_df.groupby('Account_Category')['Total_Revenue'].sum().sort_values()
revenue_by_category.plot(kind='barh', color='seagreen')
plt.title('Revenue by Account Category', fontsize=12, fontweight='bold')
plt.xlabel('Total Revenue (₹)')

# Plot 4: Dormant accounts by state
plt.subplot(3, 3, 4)
dormant_by_state = dormant_accounts.groupby('State')['Total_Revenue'].sum().sort_values().tail(10)
dormant_by_state.plot(kind='barh', color='orange')
plt.title('Dormant Accounts Revenue by State', fontsize=12, fontweight='bold')
plt.xlabel('Revenue from Dormant Accounts (₹)')

# Plot 5: Action priority
plt.subplot(3, 3, 5)
priority_counts = report_df['Action_Priority'].value_counts()
colors = ['green', 'yellow', 'orange', 'red']
priority_counts.plot(kind='pie', autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Action Priority Distribution', fontsize=12, fontweight='bold')
plt.ylabel('')

# Plot 6: Recency vs Revenue scatter
plt.subplot(3, 3, 6)
scatter = plt.scatter(report_df['Days_Since_Last_Purchase'],
                      report_df['Total_Revenue'],
                      c=report_df['Priority_Score'],
                      cmap='viridis',
                      alpha=0.6,
                      s=50)
plt.colorbar(scatter, label='Priority Score')
plt.xlabel('Days Since Last Purchase')
plt.ylabel('Total Revenue (₹)')
plt.title('Recency vs Revenue Analysis', fontsize=12, fontweight='bold')
plt.yscale('log')

# Plot 7: Dormancy periods
plt.subplot(3, 3, 7)
if 'Dormancy_Period' in dormant_accounts.columns:
    dormancy_counts = dormant_accounts['Dormancy_Period'].value_counts()
    dormancy_counts.plot(kind='bar', color='purple')
    plt.title('Dormant Accounts by Inactivity Period', fontsize=12, fontweight='bold')
    plt.xlabel('Inactivity Period')
    plt.ylabel('Number of Accounts')
    plt.xticks(rotation=45)

# Plot 8: Salesman with most dormant accounts
plt.subplot(3, 3, 8)
top_salesmen = salesman_df.sort_values('Dormant_Accounts', ascending=False).head(8)
plt.barh(top_salesmen['Salesman'], top_salesmen['Dormant_Accounts'], color='brown')
plt.title('Top Salesmen with Dormant Accounts', fontsize=12, fontweight='bold')
plt.xlabel('Number of Dormant Accounts')

# Plot 9: Products purchased by dormant accounts
plt.subplot(3, 3, 9)
top_products = dormant_products.sort_values('SalesAmount', ascending=False).head(8)
plt.barh(top_products.index, top_products['SalesAmount'], color='teal')
plt.title('Top Products by Dormant Accounts Revenue', fontsize=12, fontweight='bold')
plt.xlabel('Revenue (₹)')

plt.suptitle('COMPREHENSIVE ACCOUNT HEALTH ANALYSIS DASHBOARD', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('Account_Analysis_Dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nDashboard visualization saved to 'Account_Analysis_Dashboard.png'")

# 12. Print key insights
print("\n" + "=" * 50)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("=" * 50)

print(f"\n1. ACQUISITION STRATEGY:")
print(
    f"   • {len(one_time_buyers)} one-time buyers ({len(one_time_buyers) / len(account_analysis_reset) * 100:.1f}% of total)")
print(f"   • Generated ₹{one_time_buyers['Total_Revenue'].sum():,.0f} revenue")
print("   • RECOMMENDATION: Implement win-back campaigns with special offers for top 20 one-time buyers")

print(f"\n2. RETENTION OPPORTUNITIES:")
print(f"   • {len(low_freq_buyers)} low-frequency buyers (31.8% of accounts)")
print(f"   • RECOMMENDATION: Create loyalty programs or subscription models")

print(f"\n3. REACTIVATION PRIORITIES:")
print(f"   • {len(dormant_accounts)} dormant accounts (>90 days inactive)")
print(f"   • {len(high_value_dormant)} high-value dormant accounts (revenue > ₹{revenue_threshold:,.0f})")
print("   • RECOMMENDATION: Immediate contact campaign for high-value dormant accounts")

print(f"\n4. SALES TEAM ACTION:")
print(
    f"   • Focus on {len(report_df[report_df['Action_Priority'].isin(['Critical', 'High'])])} 'Critical' and 'High' priority accounts")
print("   • Review accounts assigned to each salesman")
print("   • Set up weekly account health reviews")

print(f"\n5. PRODUCT INSIGHTS:")
print("   • Analyze which products dormant accounts purchased")
print("   • Use this data for targeted reactivation offers")

# 13. Export for Power BI
print("\n" + "=" * 50)
print("POWER BI PREPARATION")
print("=" * 50)

# Create Power BI ready data
powerbi_data = report_df.copy()

# Add additional calculated fields for Power BI
powerbi_data['Is_Dormant'] = powerbi_data['Days_Since_Last_Purchase'] > 90
powerbi_data['Is_One_Time'] = powerbi_data['Purchase_Count'] == 1
powerbi_data['Revenue_Bucket'] = pd.cut(powerbi_data['Total_Revenue'],
                                        bins=[0, 10000, 50000, 100000, 500000, float('inf')],
                                        labels=['<10K', '10K-50K', '50K-100K', '100K-500K', '>500K'])

# Save for Power BI
powerbi_data.to_csv('PowerBI_Account_Analysis.csv', index=False)
print("Power BI ready data saved to 'PowerBI_Account_Analysis.csv'")

print("\n" + "=" * 50)
print("NEXT STEPS:")
print("=" * 50)
print("1. Review 'Account_Analysis_Reports.xlsx' for detailed insights")
print("2. Check 'Account_Analysis_Dashboard.png' for visual overview")
print("3. Use 'PowerBI_Account_Analysis.csv' for interactive dashboard")
print("4. Start with High-Priority accounts from Sheet 'High_Priority'")
print("5. Assign dormant accounts to sales team for follow-up")