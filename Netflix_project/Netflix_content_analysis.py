# ============================================================
# NETFLIX CONTENT ANALYSIS AND VISUALIZATION
# Data Science College Project
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# Plot style
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

# Make sure netflix_titles.csv is in the same folder
df = pd.read_csv("netflix_titles.csv")

print("Dataset loaded successfully!")
print("Shape of dataset:", df.shape)

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# ------------------------------------------------------------
# 3. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\nDataset Information:")
print(df.info())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nStatistical Summary:")
print(df.describe(include="all").T)

# ------------------------------------------------------------
# 4. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# Missing value percentage
missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Percentage": missing_percentage
})

print("\nMissing Value Analysis:")
print(missing_df.sort_values("Percentage", ascending=False))

# ------------------------------------------------------------
# 5. REMOVE DUPLICATES
# ------------------------------------------------------------

print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# ------------------------------------------------------------
# 6. DATA CLEANING
# ------------------------------------------------------------

# Convert date_added into datetime
df["date_added"] = pd.to_datetime(
    df["date_added"],
    errors="coerce"
)

# Extract year and month
df["added_year"] = df["date_added"].dt.year
df["added_month"] = df["date_added"].dt.month_name()

# Convert release_year to numeric
df["release_year"] = pd.to_numeric(
    df["release_year"],
    errors="coerce"
)

# Fill missing categorical values
categorical_columns = [
    "director",
    "cast",
    "country",
    "rating",
    "duration"
]

for column in categorical_columns:
    df[column] = df[column].fillna("Unknown")

# ------------------------------------------------------------
# 7. DATASET AFTER CLEANING
# ------------------------------------------------------------

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCleaned Dataset:")
print(df.head())

# ------------------------------------------------------------
# 8. CONTENT TYPE ANALYSIS
# ------------------------------------------------------------

content_type = df["type"].value_counts()

print("\nMovies vs TV Shows:")
print(content_type)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="type"
)

plt.title("Distribution of Movies and TV Shows")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 9. PIE CHART - MOVIES VS TV SHOWS
# ------------------------------------------------------------

plt.figure(figsize=(7, 7))

plt.pie(
    content_type.values,
    labels=content_type.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Movies vs TV Shows on Netflix")
plt.show()

# ------------------------------------------------------------
# 10. CONTENT ADDED BY YEAR
# ------------------------------------------------------------

content_by_year = (
    df["added_year"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    content_by_year.index,
    content_by_year.values,
    marker="o"
)

plt.title("Netflix Content Added by Year")
plt.xlabel("Year")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 11. CONTENT ADDED BY YEAR AND TYPE
# ------------------------------------------------------------

year_type = (
    df.groupby(["added_year", "type"])
    .size()
    .reset_index(name="count")
)

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=year_type,
    x="added_year",
    y="count",
    hue="type",
    marker="o"
)

plt.title("Movies vs TV Shows Added Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 12. RELEASE YEAR ANALYSIS
# ------------------------------------------------------------

release_year = (
    df["release_year"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(14, 6))

plt.plot(
    release_year.index,
    release_year.values
)

plt.title("Netflix Titles by Release Year")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 13. TOP 10 COUNTRIES
# ------------------------------------------------------------

country_data = (
    df["country"]
    .str.split(", ")
    .explode()
)

top_countries = country_data.value_counts().head(10)

print("\nTop 10 Countries:")
print(top_countries)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=top_countries.values,
    y=top_countries.index
)

plt.title("Top 10 Countries Producing Netflix Content")
plt.xlabel("Number of Titles")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 14. TOP 10 GENRES
# ------------------------------------------------------------

genre_data = (
    df["listed_in"]
    .str.split(", ")
    .explode()
)

top_genres = genre_data.value_counts().head(10)

print("\nTop 10 Genres:")
print(top_genres)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=top_genres.values,
    y=top_genres.index
)

plt.title("Top 10 Netflix Genres")
plt.xlabel("Number of Titles")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 15. GENRE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(12, 7))

top_genres.plot(
    kind="bar"
)

plt.title("Most Popular Netflix Genres")
plt.xlabel("Genre")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 16. RATINGS ANALYSIS
# ------------------------------------------------------------

ratings = df["rating"].value_counts().head(10)

print("\nNetflix Ratings:")
print(ratings)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=ratings.values,
    y=ratings.index
)

plt.title("Netflix Content by Rating")
plt.xlabel("Number of Titles")
plt.ylabel("Rating")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 17. MOVIE DURATION ANALYSIS
# ------------------------------------------------------------

movies = df[df["type"] == "Movie"].copy()

movies["duration_minutes"] = (
    movies["duration"]
    .str.extract(r"(\d+)")
    .astype(float)
)

print("\nMovie Duration Statistics:")
print(movies["duration_minutes"].describe())

plt.figure(figsize=(10, 6))

sns.histplot(
    movies["duration_minutes"].dropna(),
    bins=30,
    kde=True
)

plt.title("Distribution of Movie Durations")
plt.xlabel("Duration (Minutes)")
plt.ylabel("Number of Movies")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 18. TV SHOW SEASONS ANALYSIS
# ------------------------------------------------------------

tv_shows = df[df["type"] == "TV Show"].copy()

tv_shows["seasons"] = (
    tv_shows["duration"]
    .str.extract(r"(\d+)")
    .astype(float)
)

print("\nTV Show Seasons Statistics:")
print(tv_shows["seasons"].describe())

plt.figure(figsize=(10, 6))

sns.histplot(
    tv_shows["seasons"].dropna(),
    bins=20
)

plt.title("Distribution of TV Show Seasons")
plt.xlabel("Number of Seasons")
plt.ylabel("Number of TV Shows")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 19. TOP DIRECTORS
# ------------------------------------------------------------

directors = (
    df[df["director"] != "Unknown"]["director"]
    .str.split(", ")
    .explode()
)

top_directors = directors.value_counts().head(10)

print("\nTop 10 Directors:")
print(top_directors)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=top_directors.values,
    y=top_directors.index
)

plt.title("Top 10 Directors on Netflix")
plt.xlabel("Number of Titles")
plt.ylabel("Director")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 20. CONTENT BY MONTH
# ------------------------------------------------------------

month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]

monthly_content = (
    df["added_month"]
    .value_counts()
    .reindex(month_order)
)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=monthly_content.index,
    y=monthly_content.values
)

plt.title("Netflix Content Added by Month")
plt.xlabel("Month")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 21. TOP COUNTRIES USING TREEMAP
# ------------------------------------------------------------

top_country_df = pd.DataFrame({
    "Country": top_countries.index,
    "Count": top_countries.values
})

fig = px.treemap(
    top_country_df,
    path=["Country"],
    values="Count",
    title="Top Netflix Content Producing Countries"
)

fig.show()

# ------------------------------------------------------------
# 22. INTERACTIVE GENRE CHART
# ------------------------------------------------------------

genre_df = pd.DataFrame({
    "Genre": top_genres.index,
    "Count": top_genres.values
})

fig = px.bar(
    genre_df,
    x="Genre",
    y="Count",
    title="Top Netflix Genres",
    text="Count"
)

fig.show()

# ------------------------------------------------------------
# 23. MOVIE VS TV SHOW BY YEAR
# ------------------------------------------------------------

year_type_pivot = (
    df.groupby(["added_year", "type"])
    .size()
    .unstack(fill_value=0)
)

year_type_pivot.plot(
    kind="bar",
    figsize=(15, 7)
)

plt.title("Movies and TV Shows Added by Year")
plt.xlabel("Year")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.legend(title="Content Type")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 24. CONTENT RATING VS TYPE
# ------------------------------------------------------------

rating_type = pd.crosstab(
    df["rating"],
    df["type"]
)

top_ratings = (
    df["rating"]
    .value_counts()
    .head(8)
    .index
)

rating_type = rating_type.loc[
    rating_type.index.intersection(top_ratings)
]

rating_type.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Content Ratings by Type")
plt.xlabel("Rating")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 25. CORRELATION ANALYSIS
# ------------------------------------------------------------

numeric_df = df[
    [
        "release_year",
        "added_year"
    ]
].dropna()

correlation = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(7, 5))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 26. MOST RECENT CONTENT
# ------------------------------------------------------------

latest_content = (
    df.sort_values(
        "date_added",
        ascending=False
    )
    [
        [
            "title",
            "type",
            "date_added",
            "release_year",
            "rating"
        ]
    ]
    .head(20)
)

print("\nLatest Netflix Content:")
print(latest_content.to_string(index=False))

# ------------------------------------------------------------
# 27. CONTENT SUMMARY
# ------------------------------------------------------------

total_titles = len(df)
total_movies = len(df[df["type"] == "Movie"])
total_tv = len(df[df["type"] == "TV Show"])

print("\n================ PROJECT SUMMARY ================")

print("Total Netflix Titles :", total_titles)
print("Total Movies         :", total_movies)
print("Total TV Shows       :", total_tv)

print(
    "Movie Percentage     :",
    round((total_movies / total_titles) * 100, 2),
    "%"
)

print(
    "TV Show Percentage   :",
    round((total_tv / total_titles) * 100, 2),
    "%"
)

print(
    "Most Common Genre    :",
    top_genres.index[0]
)

print(
    "Top Producing Country:",
    top_countries.index[0]
)

print(
    "Most Common Rating   :",
    ratings.index[0]
)

# ------------------------------------------------------------
# 28. FINAL INSIGHTS
# ------------------------------------------------------------

print("\n================ KEY INSIGHTS ================")

print(
    f"1. Netflix contains {total_titles} titles "
    f"after data cleaning."
)

print(
    f"2. Movies account for "
    f"{round((total_movies / total_titles) * 100, 2)}% "
    f"of the dataset."
)

print(
    f"3. TV Shows account for "
    f"{round((total_tv / total_titles) * 100, 2)}% "
    f"of the dataset."
)

print(
    f"4. The most common genre is "
    f"{top_genres.index[0]}."
)

print(
    f"5. The country with the highest number of "
    f"titles is {top_countries.index[0]}."
)

print(
    f"6. The most common content rating is "
    f"{ratings.index[0]}."
)

print(
    "7. Netflix has significantly expanded its "
    "content library over the years."
)

print(
    "8. Movie duration and TV-show season analysis "
    "helps understand Netflix's content structure."
)

print("\n================================================")
print("PROJECT COMPLETED SUCCESSFULLY!")
print("================================================")