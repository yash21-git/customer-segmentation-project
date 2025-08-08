# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_stata(r'/Users/yash/Desktop/HCES(2023-24)/clean/hces2024inf.dta')

# %%
var_counts = df.notnull().sum().reset_index()
var_counts.columns = ['Variable', 'Non-Missing Count']

# Display the result
print(var_counts)

# %%
df.describe()


# %%
# Define the variables to be used for correlation analysis
vars= [
    "mpce", "E_com_index", "sex_hoh", "age_hoh", "source_cooking", "source_lighting",
    "type_house", "has_land", "caste", "hoh_religion", "hh_type", "is_rent",
    "employed_annual", "hh_size", "used_ration", "is_hhmem_pmjay",
    "has_pmgky", "max_edu_level"
]
df_corr = df[vars]
df_corr_clean = df_corr.dropna()
corr_matrix = df_corr_clean.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True,
            linewidths=0.5, cbar_kws={"label": "Correlation Coefficient"})
plt.title("Correlation Heatmap – HCES 2023-24 Variables", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# %%
# 1. DIGITAL ADOPTION SCORE
online_cols = [col for col in df.columns if col.startswith("online_")]

# Convert to numeric
df[online_cols] = df[online_cols].apply(pd.to_numeric, errors='coerce')

# Compute score
df["digital_adoption_score"] = df[online_cols].mean(axis=1)


# %%
# 2. ASSET MODERNITY INDEX
asset_cols = [
    "has_tv", "has_fridge", "has_mobile", "has_laptop", "has_bike", "has_car", "has_truck",
    "has_radio", "has_bicycle", "has_washing_machine", "has_ac", "has_house", "has_land", "has_animal_cart"
]
existing_asset_cols = [col for col in asset_cols if col in df.columns]
df["asset_modernity_index"] = df[existing_asset_cols].mean(axis=1)

# %%
# 3. SUBSIDY DEPENDENCE INDEX
subsidy_vars = ["is_hhmem_pmjay", "has_pmgky", "used_ration"]
df["subsidy_dependence_index"] = df[subsidy_vars].sum(axis=1) / df["hh_size"]


# %%
# 4. VULNERABILITY SCORE
# Poverty proxies: low MPCE, no land, is_rent, high inflation
df["low_mpce"] = (df["mpce"] < df["mpce"].quantile(0.25)).astype(int)
df["no_land"] = (df["has_land"] == 0).astype(int)
df["renting"] = (df["is_rent"] == 1).astype(int)
df["high_inflation"] = (df["inflation_rate"] > df["inflation_rate"].median()).astype(int)

df["vulnerability_score"] = df[["low_mpce", "no_land", "renting", "high_inflation"]].mean(axis=1)


# %%
# Preview generated variables
df_subset = df[[
    "digital_adoption_score",
    "asset_modernity_index",
    "subsidy_dependence_index",
    "vulnerability_score"
]].describe()

# %%
df_subset

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Select features for clustering
segmentation_vars = [
    "log_mpce",  # Monthly per capita expenditure
    "digital_adoption_score",
    "asset_modernity_index",
    "subsidy_dependence_index",
    "vulnerability_score",
    "hh_size",
    "inflation_rate"
]

# Drop missing values and scale features
cluster_data = df[segmentation_vars].dropna().copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_data)

# %%
# Elbow method to find optimal number of clusters
wcss = []
K_range = range(2, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(K_range, wcss, marker='o')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.show()


# %%
# Choose optimal k (e.g., 3) and fit KMeans
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# %%
# Cluster Profiling
cluster_summary = df.groupby("cluster")[segmentation_vars].mean().round(2)
display(cluster_summary)

# Assign human-readable labels to clusters
cluster_map = {
    0: "Subsidy-Dependent Rural",
    1: "Affluent Urban",
    2: "Digitally Progressive Middle"
}
df['segment_label'] = df['cluster'].map(cluster_map)


# %% [markdown]
# cluster_summary

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Sample data: Replace this with your real cluster-wise mean DataFrame
cluster_means = df.groupby("cluster")[[
    "log_mpce", "digital_adoption_score", "asset_modernity_index",
    "subsidy_dependence_index", "vulnerability_score", "hh_size"
]].mean()

# Define variables and angles
categories = cluster_means.columns.tolist()
num_vars = len(categories)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # repeat the first angle to close the loop

# Create polar plot
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

# Plot each cluster
for idx, row in cluster_means.iterrows():
    values = row.tolist()
    values += values[:1]  # repeat first value to close the loop
    ax.plot(angles, values, label=f"Cluster {idx}")
    ax.fill(angles, values, alpha=0.25)

# Set the labels for each axis
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Optional: Add title and legend
plt.title('Radar Chart of Cluster Characteristics')
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

plt.tight_layout()
plt.show()


# %%
import seaborn as sns

sns.barplot(data=df, x='segment_label', y='log_mpce')
plt.title('Average log(MPCE) by Segment')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# %%
import scipy.stats as stats

grouped = [group['log_mpce'].values for name, group in df.groupby('cluster')]
f_stat, p_val = stats.f_oneway(*grouped)
print(f"ANOVA F-statistic: {f_stat:.3f}, p-value: {p_val:.3f}")


# %%
for c in df['cluster'].unique():
    desc = df[df['cluster'] == c][[
        'log_mpce', 'digital_adoption_score', 'asset_modernity_index',
        'subsidy_dependence_index', 'vulnerability_score', 'hh_size'
    ]].mean().to_dict()
    
    prompt = f"""Generate a one-paragraph summary for Cluster {c} based on:
    - MPCE: {desc['log_mpce']:.2f}
    - Digital Adoption Score: {desc['digital_adoption_score']:.2f}
    - Asset Modernity Index: {desc['asset_modernity_index']:.2f}
    - Subsidy Dependence Index: {desc['subsidy_dependence_index']:.2f}
    - Vulnerability Score: {desc['vulnerability_score']:.2f}
    - Household Size: {desc['hh_size']:.2f}
    """
    print(prompt)



