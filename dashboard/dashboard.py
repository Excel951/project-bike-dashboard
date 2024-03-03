import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import datetime
import os
from sklearn.cluster import KMeans

sns.set_theme(style='dark')

def create_monthly_rentals_df(day_df):
    q1_df = day_df
    q1_df['year'] = q1_df['dteday'].dt.year
    q1_df = q1_df.groupby([q1_df['dteday'].dt.strftime('%Y-%m'), 'year']).agg({
        'cnt': 'sum'
    })
    q1_df.index = q1_df.index.set_names(['month_year', 'year'])
    q1_df.reset_index(inplace=True)
    q1_df['month_year'] = pd.to_datetime(q1_df['month_year'])
    q1_df.rename(columns={
        'cnt': 'total_rentals'
    }, inplace=True)
    q1_df.sort_values(by='month_year', ascending=False)
    
    return q1_df
    
def create_seasons_rentals_df(day_df):
    q2_df = day_df
    q2_df['season_name'] = q2_df.season.apply(lambda x: "Springer" if x == 1 else ("Summer" if x == 2 else ("Fall" if x == 3 else "Winter")))
    season_most_liked_df = q2_df.groupby(by='season_name').cnt.sum().sort_values(ascending=False).reset_index()

    return season_most_liked_df
    
def create_clustering_characteristics_df(hour_df):
    q3_df = hour_df

    kmeans_clustering_parameter = q3_df[['cnt', 'hr']]

    kmeans_clustering = KMeans(n_clusters=3)
    kmeans_clustering.fit(kmeans_clustering_parameter)

    q3_df['cluster'] = kmeans_clustering.labels_
    
    return q3_df

script_dir = os.path.dirname(__file__)
day_csv_path = os.path.join(script_dir, 'day.csv')
hour_csv_path = os.path.join(script_dir, 'hour.csv')

day_df = pd.read_csv(day_csv_path)
hour_df = pd.read_csv(hour_csv_path)

day_df['dteday'] = pd.to_datetime(day_df['dteday'])
hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])

min_date = day_df['dteday'].min()
max_date = day_df['dteday'].max()

with st.sidebar:
    st.header("Hello,\nWelcome to Bike Dashboard!")
    
    start_date, end_date = st.date_input(
        label='Pilih rentang waktu',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )
    
day_main_df = day_df[(day_df['dteday'] >= str(start_date)) & (day_df['dteday'] <= str(end_date))]
hour_main_df = hour_df[(hour_df['dteday'] >= str(start_date)) & (hour_df['dteday'] <= str(end_date))]

monthly_rentals_df = create_monthly_rentals_df(day_main_df)
seasons_rentals_df = create_seasons_rentals_df(day_main_df)
clustering_character_df = create_clustering_characteristics_df(hour_main_df)

st.header("Bike Dasboard :bike:")

# code for monthly rentals
# =================================================================
st.subheader('Monthly Bike Rentals')

with st.container():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly_rentals_df['month_year'],monthly_rentals_df['total_rentals'], marker='o', linewidth=2, color='#90CAF9')
    plt.xticks(fontsize=10) 
    plt.yticks(fontsize=10)
    ax.set_ylim(0, monthly_rentals_df['total_rentals'].max() * 1.1)
    st.pyplot(fig=fig)

# =================================================================

# code for seasons rentals
# =================================================================
st.subheader("Season Most Favored by Customers")
colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

with st.container():
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='cnt', y='season_name', data=seasons_rentals_df.head(), palette=colors, ax=ax)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    for index, value in enumerate(seasons_rentals_df.head()['cnt']):
        plt.text(value, index, str(value), ha='right', va='center')
    st.pyplot(fig=fig)
# =================================================================

# code for clustering 
# =================================================================
st.subheader('Clustering of Bike Rental Characteristics')

with st.container():
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster in sorted(clustering_character_df['cluster'].unique()):
        cluster_data = clustering_character_df[clustering_character_df['cluster'] == cluster]
        plt.scatter(cluster_data['hr'], cluster_data['cnt'], label=f'Kelompok Customer {cluster+1}')

    ax.set_xlabel('Hour')
    plt.legend()
    st.pyplot(fig)
# =================================================================

st.caption('by Abigail Excelsis Deo')