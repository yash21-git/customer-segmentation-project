# customer-segmentation-project
This project uses data from households across India to group them into different categories based on how they spend money, use digital tools, own assets, and depend on government support.

**Consumer Segmentation Using HCES 2023–24 Data**

This project uses unsupervised machine learning (K-Means Clustering) on the nationally representative HCES 2023–24 dataset to segment Indian households into distinct consumer groups. By engineering interpretable features such as Digital Adoption, Asset Ownership, Subsidy Dependence, and a Vulnerability Score, we aim to offer actionable insights for welfare targeting, financial inclusion, and digital strategy.

**Data Overview**

1.Source: Household Consumption and Expenditure Survey (HCES) 2023–24-https://microdata.gov.in/NADA/index.php/catalog/237

2.Coverage: ~270,000+ households across rural and urban India

3.Collected By: Ministry of Statistics and Programme Implementation (MoSPI), Government of India

**Why HCES Data?**
It captures detailed household-level socio-economic information.
Allows construction of custom indices to reflect consumption patterns, digital access, asset holdings, and vulnerability.
About the Data (HCES 2023–24)
HCES = Household Consumption and Expenditure Survey
Conducted by India’s Ministry of Statistics and Programme Implementation (MoSPI)
Covers thousands of Indian households
Includes details like:

Monthly spending (food, fuel, transport, etc.)

Use of electricity, LPG, and internet

Asset ownership (TV, fridge, phone, land, etc.)

Government scheme usage (like ration cards, PMGKY, PMJAY)

**Objective**

Segment households based on multiple indicators to identify groups that differ in digital inclusion, asset base, and dependency on government subsidies.

Make the segmentation interpretable and usable for policy and program design.

**What did I do with the data?**
Step 1: Cleaned and Prepared the Data
Merged 15+ files from the official survey into one dataset

Created new binary variables like has_fridge, uses_lpg, online_payments, etc.

Step 2: Built Indices (Scorecards)
I transformed over 100 variables into meaningful scores:
| Index                        | What it Measures                     | Example Inputs                           |
| ---------------------------- | ------------------------------------ | ---------------------------------------- |
| **Digital Adoption Score**   | How tech-friendly a household is     | Internet use, online banking, e-commerce |
| **Asset Modernity Index**    | How modern their possessions are     | Refrigerator, AC, vehicle                |
| **Subsidy Dependence Index** | How much they rely on govt help      | Free rations, LPG subsidy, PMJAY         |
| **Vulnerability Score**      | How economically vulnerable they are | Irregular job, poor housing, no toilet   |

Step 3: Applied K-Means Clustering
K-Means is a machine learning method to group similar households together based on the scores above + spending and household size.

I discovered 4 distinct clusters or “household types.”

Step 4: Analyzed and Visualized the Results
Ran statistical tests (t-tests, ANOVA) to confirm differences between clusters.

Used radar charts and bar plots to visualize each cluster’s profile.


**Methodology**

Feature Engineering: Created 4 core indices:

E-com Index: Reflects digital commerce and platform usage.

Wealth Index: Based on durable goods and asset ownership.

Subsidy Score: Indicates reliance on key government schemes.

Vulnerability Score: Combines housing conditions, sanitation, rent, etc.

Clustering Algorithm: K-Means Clustering with K=4 (based on Elbow & Silhouette)

**Statistical Testing:**

ANOVA and t-tests to confirm inter-cluster differences

Summary stats for MPCE, digital access, asset levels, etc.

**Visualization:**

Radar plots to show the profile of each cluster

Bar charts to compare MPCE and index values by cluster

**Key concpets**
| Term                | Layman Explanation                                                |
| ------------------- | ----------------------------------------------------------------- |
| MPCE                | Monthly spending per person (how much a family spends per head)   |
| K-Means Clustering  | Grouping similar households without knowing their type in advance |
| Index               | A score created using multiple variables to summarize a concept   |
| Statistical Testing | Checking if differences between groups are real or random         |
| Radar Plot          | A web-like chart that shows multiple scores in one view           |


**What did I Find**
Each cluster represnets a unique story:
| Cluster   | Summary                                                                         |
| --------- | ------------------------------------------------------------------------------- |
| Cluster 0 | Moderate income, low tech use, moderate subsidy use — mostly in transition      |
| Cluster 1 | Low MPCE, high subsidy dependence, digitally excluded — at-risk group           |
| Cluster 2 | High digital use, high asset ownership, low subsidy — self-reliant & modern     |
| Cluster 3 | Large households, moderate digital adoption, average vulnerability — mixed type |
These categories highlight the diversity of Indian households — which can't be understood from income alone.

**Why did I do this?**
Understanding how different types of households behave helps policymakers and financial institutions (like the RBI) design better welfare programs, target digital services, and make informed decisions about financial inclusion.

For example:
Some households are digitally disconnected but depend on subsidies.
Others are tech-savvy, self-reliant, and need fewer government benefits.
This kind of insight can guide targeted interventions, saving money and improving impact.

**Role of Machine Learning & Generative AI**
This project leverages both Machine Learning (ML) and Generative AI (GenAI) to draw meaningful insights from HCES 2023–24 microdata:

Machine Learning for Segmentation
Applied K-Means Clustering to group Indian households into 4 distinct segments based on key dimensions: digital adoption, asset ownership, subsidy dependence, and vulnerability score.
Engineered features using 100+ raw variables to build interpretable indices.
Used ANOVA and t-tests to validate statistical differences between the clusters, ensuring reliable insights.

Generative AI for Interpretation
Used GenAI (ChatGPT) to interpret and summarize technical results in human-readable narratives.
Made it easier to describe each household group clearly and communicate insights with non-technical stakeholders.

Why It Matters
Combining ML’s ability to detect patterns with GenAI’s storytelling capabilities, this project delivers:
Actionable, data-driven segmentation
Easy-to-understand summaries for decision-making
A template for combining advanced analytics with explainability


