import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import json
import folium
from streamlit_folium import folium_static
import statsmodels.api as sm
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("Analiza Avansată a Pieței Auto")

st.sidebar.header("Opțiuni de Analiză")
analiza_selectata = st.sidebar.selectbox(
    "Selectați tipul de analiză:",
    ["Dashboard General", "Analiză Statistică Avansată",
     "Prelucrare și Transformare Date", "Analiză Temporală", "Grupare și Segmentare",
     "Modele Machine Learning", "Regresie Liniară Statsmodels"]
)

@st.cache_data
def incarcare_date():
    df = pd.read_csv("car_prices.csv", header=None)
    df.columns = ['Year', 'Make', 'Model', 'Version', 'Type', 'Transmission',
                 'VIN', 'Location', 'Owners', 'Mileage', 'ExteriorColor',
                 'InteriorColor', 'Dealer', 'ListPrice', 'FinalPrice', 'Date']

    numeric_cols = ['Year', 'Owners', 'Mileage', 'ListPrice', 'FinalPrice']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')


    current_year = datetime.now().year
    df['Age'] = current_year - df['Year']
    df['Discount'] = df['ListPrice'] - df['FinalPrice']
    df['DiscountPercentage'] = (df['Discount'] / df['ListPrice']) * 100

    return df

@st.cache_data
def incarcare_harta():
    try:
        romania_map = gpd.read_file("romania-counties.geojson")
        return romania_map
    except Exception as e:
        st.warning("Fișierul GeoJSON pentru harta României nu a fost găsit.")
        return None

try:
    df = incarcare_date()
    if len(df) == 0:
        st.error("Nu s-au putut încărca date valide din fișierul CSV.")
        st.stop()
except Exception as e:
    st.error(f"Eroare la încărcarea datelor: {str(e)}")
    st.stop()

romania_map = incarcare_harta()


def tratare_date_avansata(df):
    df_clean = df.copy()

    numeric_cols = ['Year', 'Owners', 'Mileage', 'ListPrice', 'FinalPrice']
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_clean[~((df_clean[col] < (Q1 - 1.5 * IQR)) |
                             (df_clean[col] > (Q3 + 1.5 * IQR)))]
    return df_clean

def analiza_statistica_avansata(df):
    stats_dict = {
        'Statistici de bază': df[['FinalPrice', 'Mileage', 'Age']].describe(),
        'Skewness': df[['FinalPrice', 'Mileage', 'Age']].skew(),
        'Kurtosis': df[['FinalPrice', 'Mileage', 'Age']].kurtosis(),
        'Corelații': df[['FinalPrice', 'Mileage', 'Age', 'DiscountPercentage']].corr()
    }
    return stats_dict

def create_dashboard(df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Preț Mediu", f"{df['FinalPrice'].mean():,.0f} €")
    with col2:
        st.metric("Număr Total Mașini", len(df))
    with col3:
        st.metric("Discount Mediu", f"{df['DiscountPercentage'].mean():.1f}%")
    with col4:
        st.metric("Vârsta Medie", f"{df['Age'].mean():.1f} ani")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Distribuția Prețurilor pe Mărci")
        fig = px.box(df, x='Make', y='FinalPrice',
                    title='Distribuția Prețurilor pe Mărci')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("### Top Mărci după Număr de Mașini")
        marca_counts = df['Make'].value_counts().head(10)
        fig = px.bar(x=marca_counts.index, y=marca_counts.values,
                    title='Top 10 Mărci')
        st.plotly_chart(fig, use_container_width=True)

def create_map(df):

    try:
        with open('romania-counties.geojson', 'r', encoding='utf-8') as f:
            counties_data = json.load(f)
    except Exception as e:
        st.error("Nu s-a putut încărca fișierul GeoJSON")
        return None


    m = folium.Map(location=[45.9432, 24.9668], zoom_start=7)
    county_stats = df.groupby('Location').agg({
        'FinalPrice': 'mean',
        'VIN': 'count'
    }).reset_index()



    for feature in counties_data['features']:
        county_name = feature['properties']['name']
        coords = feature['geometry']['coordinates']



        county_data = county_stats[county_stats['Location'] == county_name]
        if not county_data.empty:
            avg_price = county_data['FinalPrice'].iloc[0]
            car_count = county_data['VIN'].iloc[0]

            folium.CircleMarker(
                location=[coords[1], coords[0]],
                radius=10,
                popup=f"{county_name}<br>Preț mediu: {avg_price:,.0f}€<br>Număr mașini: {car_count}",
                color='red',
                fill=True
            ).add_to(m)

    return m

def analiza_temporala(df):
    df_temporal = df.groupby('Date').agg({
        'FinalPrice': 'mean',
        'VIN': 'count',
        'DiscountPercentage': 'mean'
    }).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_temporal['Date'], y=df_temporal['FinalPrice'],
                            name='Preț Mediu'))
    fig.add_trace(go.Scatter(x=df_temporal['Date'], y=df_temporal['DiscountPercentage'],
                            name='Discount Mediu %', yaxis='y2'))

    fig.update_layout(
        title='Evoluția Prețurilor și Discounturilor în Timp',
        yaxis=dict(title='Preț Mediu'),
        yaxis2=dict(title='Discount %', overlaying='y', side='right')
    )

    st.plotly_chart(fig)


if analiza_selectata == "Dashboard General":
    st.write("## Dashboard General al Pieței Auto")
    create_dashboard(df)

elif analiza_selectata == "Analiză Statistică Avansată":
    st.write("## Analiză Statistică Avansată")
    df_curat = tratare_date_avansata(df)


    st.write("### Statistici Descriptive")
    st.write(df_curat[['FinalPrice', 'Mileage', 'Age']].describe())

    st.write("### Matrice de Corelație")
    corr = df_curat[['FinalPrice', 'Mileage', 'Age', 'DiscountPercentage']].corr()



    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=corr.columns,
        y=corr.columns,
        text=corr.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorscale='RdBu'
    ))

    fig.update_layout(
        title='Matricea de Corelație',
        width=600,
        height=600
    )

    st.plotly_chart(fig)



    col1, col2 = st.columns(2)
    with col1:
        st.write("### Distribuția Prețurilor")
        fig = px.histogram(df_curat, x='FinalPrice', nbins=50)
        st.plotly_chart(fig)

    with col2:
        st.write("### Relația Preț-Kilometraj")
        fig = px.scatter(df_curat, x='Mileage', y='FinalPrice',
                        trendline='ols')
        st.plotly_chart(fig)

elif analiza_selectata == "Analiză Temporală":
    st.write("## Analiza Temporală a Pieței")
    analiza_temporala(df)

elif analiza_selectata == "Prelucrare și Transformare Date":
    st.write("## Prelucrare și Transformare Date")
    le = LabelEncoder()
    df_encoded = df.copy()
    categorical_cols = ['Make', 'Model', 'Type', 'Transmission']

    for col in categorical_cols:
        df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col])



    scaler = StandardScaler()
    numeric_cols = ['FinalPrice', 'Mileage', 'Age']

    for col in numeric_cols:
        df_encoded[f'{col}_scaled'] = scaler.fit_transform(df_encoded[[col]])

    st.write("### Date Procesate (primele 10 rânduri)")
    st.dataframe(df_encoded.head(10))


    st.write("### Distribuția Datelor Scalate")
    fig = px.box(df_encoded,
                 y=[col for col in df_encoded.columns if col.endswith('_scaled')])
    st.plotly_chart(fig)

elif analiza_selectata == "Grupare și Segmentare":
    st.write("## Analiză pe Grupuri")

    group_by = st.selectbox("Grupează după:", ['Make', 'Type', 'Transmission'])
    group_stats = df.groupby(group_by).agg({
        'FinalPrice': ['mean', 'median', 'count', 'std'],
        'Mileage': 'mean',
        'DiscountPercentage': 'mean'
    }).round(2)

    st.write("### Statistici pe Grupuri")
    st.write(group_stats)
    st.write("### Comparație Prețuri pe Grupuri")
    fig = px.box(df, x=group_by, y='FinalPrice', color=group_by)
    st.plotly_chart(fig)

elif analiza_selectata == "Modele Machine Learning":
    st.write("## Modele Machine Learning")


    df_ml = df.copy()
    df_ml = df_ml.dropna(subset=['FinalPrice', 'Mileage', 'Age'])
    features = df_ml[['FinalPrice', 'Mileage', 'Age']]

    st.subheader("Clusterizare cu KMeans")


    num_clusters = st.slider("Selectează numărul de clustere", 2, 6, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    df_ml['Cluster'] = kmeans.fit_predict(features_scaled)

    fig = px.scatter(df_ml, x='FinalPrice', y='Mileage',
                     color=df_ml['Cluster'].astype(str),
                     symbol='Cluster',
                     title="Clusterizare Mașini")

    st.plotly_chart(fig)

    st.dataframe(df_ml[['Make', 'Model', 'FinalPrice', 'Mileage', 'Age', 'Cluster']].head(100))

    st.markdown("---")

    st.subheader("Regresie Logistică: Predictia Discountului Mare")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    df_ml['HighDiscount'] = (df_ml['DiscountPercentage'] > df_ml['DiscountPercentage'].median()).astype(int)

    X = df_ml[['FinalPrice', 'Mileage', 'Age']]
    y = df_ml['HighDiscount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("### Matrice de Confuzie")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                    labels=dict(x="Predict", y="Actual"), x=['Nu', 'Da'], y=['Nu', 'Da'])
    st.plotly_chart(fig)

    st.write("### Raport de Clasificare")
    st.text(classification_report(y_test, y_pred))

elif analiza_selectata == "Regresie Liniară Statsmodels":


    st.write("## Regresie Liniară Multiplă (statsmodels)")

    df_reg = df.dropna(subset=['FinalPrice', 'Mileage', 'Age'])
    X = df_reg[['Mileage', 'Age']]
    y = df_reg['FinalPrice']

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    st.write("### Rezumatul Modelului")
    st.text(model.summary())

    st.write("### Comparație Valori Reale vs. Prezise")
    df_reg['PredictedPrice'] = model.predict(X)
    fig = px.scatter(df_reg, x='FinalPrice', y='PredictedPrice', trendline='ols',
                     labels={'FinalPrice': 'Preț Real', 'PredictedPrice': 'Preț Prezis'})
    fig.add_shape(type="line", x0=df_reg['FinalPrice'].min(), y0=df_reg['FinalPrice'].min(),
                  x1=df_reg['FinalPrice'].max(), y1=df_reg['FinalPrice'].max(),
                  line=dict(color="Red", dash="dash"))
    st.plotly_chart(fig)

    st.write("### Primele 10 Predicții")
    st.dataframe(df_reg[['FinalPrice', 'PredictedPrice', 'Mileage', 'Age']].head(10))
with st.sidebar:
    st.write("### Informații Dataset")
    st.write(f"Număr total înregistrări: {len(df)}")
    st.write(f"Număr mărci unice: {df['Make'].nunique()}")
    st.write(f"Număr modele unice: {df['Model'].nunique()}")

    if st.checkbox("Afișează statistici avansate"):
        st.write("### Statistici avansate")
        st.write(f"Preț mediu: {df['FinalPrice'].mean():,.2f} €")
        st.write(f"Kilometraj mediu: {df['Mileage'].mean():,.0f} km")
        st.write(f"Vârsta medie: {df['Age'].mean():.1f} ani")
        st.write(f"Discount mediu: {df['DiscountPercentage'].mean():.1f}%")
