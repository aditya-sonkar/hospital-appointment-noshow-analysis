import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_analysis():
    print("Loading data...")
    # Load dataset
    df = pd.read_csv('../data/KaggleV2-May-2016.csv')

    print("1. CLEANING DATA...")
    # Fix column name typos
    df.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap', 'No-show': 'No_show'}, inplace=True)

    # Convert ScheduledDay and AppointmentDay to datetime
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

    # Remove invalid rows (e.g. negative Age)
    df = df[df['Age'] >= 0]

    # Add an AgeGroup column (0-18, 19-40, 41-60, 60+)
    bins = [-1, 18, 40, 60, 150]
    labels = ['0-18', '19-40', '41-60', '60+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

    # Add a WaitingDays column (AppointmentDay minus ScheduledDay)
    # We normalize to date to ignore the time part of ScheduledDay
    df['WaitingDays'] = (df['AppointmentDay'].dt.normalize() - df['ScheduledDay'].dt.normalize()).dt.days
    # Remove negative waiting days (erroneous data where appointment is before scheduled date)
    df = df[df['WaitingDays'] >= 0]

    # Save as cleaned_appointments.csv
    df.to_csv('../data/cleaned_appointments.csv', index=False)
    print("Cleaned data saved to '../data/cleaned_appointments.csv'")

    print("\n2. ANALYZING DATA...")
    # Convert No_show to binary for easy mean calculation (1 for No, 0 for Yes wait, No means they DID show up, Yes means they DID NOT show up)
    # Kaggle dataset No-show: 'Yes' = did not show up, 'No' = showed up
    df['No_show_binary'] = df['No_show'].apply(lambda x: 1 if x == 'Yes' else 0)

    insights = []

    # Overall no-show rate (%)
    overall_rate = df['No_show_binary'].mean() * 100
    insights.append(f"Overall No-Show Rate: {overall_rate:.2f}%")

    # No-show rate by SMS_received
    sms_rate = df.groupby('SMS_received')['No_show_binary'].mean() * 100
    insights.append(f"No-Show Rate without SMS (0): {sms_rate[0]:.2f}%")
    insights.append(f"No-Show Rate with SMS (1): {sms_rate[1]:.2f}%")

    # No-show rate by AgeGroup
    age_rate = df.groupby('AgeGroup')['No_show_binary'].mean() * 100
    for group, rate in age_rate.items():
        insights.append(f"No-Show Rate for Age {group}: {rate:.2f}%")

    # Top 10 Neighbourhoods by no-show rate (min. 100 appointments)
    neighborhood_counts = df['Neighbourhood'].value_counts()
    valid_neighborhoods = neighborhood_counts[neighborhood_counts >= 100].index
    neighborhood_rate = df[df['Neighbourhood'].isin(valid_neighborhoods)].groupby('Neighbourhood')['No_show_binary'].mean() * 100
    top_10_neighborhoods = neighborhood_rate.sort_values(ascending=False).head(10)
    insights.append("Top 10 Neighbourhoods with Highest No-Show Rates (min 100 appts):")
    for n, r in top_10_neighborhoods.items():
        insights.append(f"  - {n}: {r:.2f}%")

    # No-show rate by day of week
    df['DayOfWeek'] = df['AppointmentDay'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_rate = df.groupby('DayOfWeek')['No_show_binary'].mean() * 100
    day_rate = day_rate.reindex(day_order).dropna()
    insights.append("No-Show Rate by Day of Week:")
    for d, r in day_rate.items():
        insights.append(f"  - {d}: {r:.2f}%")

    # Save insights
    with open('../outputs/summary_insights.csv', 'w') as f:
        for line in insights:
            print(line)
            f.write(line + "\n")
    print("\nInsights saved to '../outputs/summary_insights.csv'")

    print("\n3. BUILDING CHARTS...")
    sns.set_theme(style="whitegrid")
    
    # Chart 1: SMS Received
    plt.figure(figsize=(6, 4))
    sns.barplot(x=sms_rate.index, y=sms_rate.values, palette="viridis")
    plt.title('No-Show Rate by SMS Received')
    plt.ylabel('No-Show Rate (%)')
    plt.xlabel('Received SMS (0 = No, 1 = Yes)')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('../outputs/chart1_sms.png')
    plt.close()

    # Chart 2: Age Group
    plt.figure(figsize=(8, 5))
    sns.barplot(x=age_rate.index, y=age_rate.values, palette="muted")
    plt.title('No-Show Rate by Age Group')
    plt.ylabel('No-Show Rate (%)')
    plt.xlabel('Age Group')
    plt.tight_layout()
    plt.savefig('../outputs/chart2_age.png')
    plt.close()

    # Chart 3: Top 10 Neighbourhoods
    plt.figure(figsize=(10, 6))
    sns.barplot(y=top_10_neighborhoods.index, x=top_10_neighborhoods.values, palette="rocket")
    plt.title('Top 10 Neighbourhoods by No-Show Rate')
    plt.xlabel('No-Show Rate (%)')
    plt.ylabel('Neighbourhood')
    plt.tight_layout()
    plt.savefig('../outputs/chart3_neighbourhoods.png')
    plt.close()

    # Chart 4: Day of Week
    plt.figure(figsize=(8, 5))
    sns.barplot(x=day_rate.index, y=day_rate.values, palette="pastel")
    plt.title('No-Show Rate by Day of Week')
    plt.ylabel('No-Show Rate (%)')
    plt.xlabel('Day of Week')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../outputs/chart4_dayofweek.png')
    plt.close()
    
    print("Charts saved as PNG files (../outputs/chart1_sms.png, etc.)")

if __name__ == "__main__":
    run_analysis()
