# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis (EDA) Module
======================================
Comprehensive visualization and analysis of video game sales data.

What was done:
    - Sales distribution analysis
    - Platform-wise comparison
    - Genre-wise trends
    - Regional sales comparison
    - Correlation analysis
    - Publisher analysis
    - Temporal trends

Key methods:
    - Seaborn countplot, barplot, heatmap
    - Matplotlib for customization
    - GroupBy aggregations

Attributes involved:
    - Genre, Platform, Publisher (categorical)
    - NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales (numerical)
    - Year (temporal)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import (
    FIGURE_SIZE_LARGE, 
    FIGURE_SIZE_XLARGE, 
    FIGURE_SIZE_MEDIUM,
    SALES_COLUMNS,
    REGIONAL_SALES_COLUMNS
)

# Set visualization style
sns.set_style('whitegrid')


# =============================================================================
# GENRE ANALYSIS
# =============================================================================

def plot_genre_distribution(df):
    """Plot the distribution of games by genre."""
    plt.figure(figsize=FIGURE_SIZE_LARGE)
    sns.countplot(x="Genre", data=df, order=df['Genre'].value_counts().index)
    plt.xticks(rotation=90)
    plt.title("Game Distribution by Genre")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_genre_sales(df):
    """Plot global sales by genre."""
    data_genre = df.groupby(by=['Genre'])['Global_Sales'].sum().reset_index()
    data_genre = data_genre.sort_values(by=['Global_Sales'], ascending=False)
    
    plt.figure(figsize=FIGURE_SIZE_LARGE)
    sns.barplot(x="Genre", y="Global_Sales", data=data_genre)
    plt.xticks(rotation=90)
    plt.title("Global Sales by Genre")
    plt.xlabel("Genre")
    plt.ylabel("Global Sales (Millions)")
    plt.tight_layout()
    plt.show()


# =============================================================================
# YEARLY ANALYSIS
# =============================================================================

def plot_yearly_releases(df):
    """Plot game releases by year."""
    plt.figure(figsize=FIGURE_SIZE_LARGE)
    order = df.groupby(by=['Year'])['Platform'].count().sort_values(ascending=False).index
    sns.countplot(x="Year", data=df, order=order)
    plt.xticks(rotation=90)
    plt.title("Game Releases by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Games")
    plt.tight_layout()
    plt.show()


def plot_yearly_sales(df):
    """Plot global sales by year."""
    data_year = df.groupby(by=['Year'])['Global_Sales'].sum().reset_index()
    
    plt.figure(figsize=FIGURE_SIZE_LARGE)
    sns.barplot(x="Year", y="Global_Sales", data=data_year)
    plt.xticks(rotation=90)
    plt.title("Global Sales by Year")
    plt.xlabel("Year")
    plt.ylabel("Global Sales (Millions)")
    plt.tight_layout()
    plt.show()


def plot_top_years_by_genre(df, top_n=5):
    """Plot top years by game releases, colored by genre."""
    plt.figure(figsize=FIGURE_SIZE_XLARGE)
    top_years = df.Year.value_counts().iloc[:top_n].index
    sns.countplot(x="Year", data=df, hue='Genre', order=top_years)
    plt.xticks(size=16, rotation=90)
    plt.title(f"Top {top_n} Years - Game Releases by Genre")
    plt.tight_layout()
    plt.show()


# =============================================================================
# PLATFORM ANALYSIS
# =============================================================================

def plot_platform_sales(df):
    """Plot global sales by platform."""
    data_platform = df.groupby(by=['Platform'])['Global_Sales'].sum().reset_index()
    data_platform = data_platform.sort_values(by=['Global_Sales'], ascending=False)
    
    plt.figure(figsize=FIGURE_SIZE_LARGE)
    sns.barplot(x="Platform", y="Global_Sales", data=data_platform)
    plt.xticks(rotation=90)
    plt.title("Global Sales by Platform")
    plt.xlabel("Platform")
    plt.ylabel("Global Sales (Millions)")
    plt.tight_layout()
    plt.show()


# =============================================================================
# REGIONAL COMPARISON
# =============================================================================

def plot_regional_comparison_by_genre(df):
    """Plot regional sales comparison heatmap by genre."""
    comp_genre = df[['Genre'] + REGIONAL_SALES_COLUMNS]
    comp_map = comp_genre.groupby(by=['Genre']).sum()
    
    plt.figure(figsize=FIGURE_SIZE_LARGE)
    sns.set(font_scale=1)
    sns.heatmap(comp_map, annot=True, fmt='.1f', cmap='Blues')
    plt.title("Regional Sales by Genre (Heatmap)")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_regional_comparison_bar(df):
    """Plot regional sales comparison bar chart by genre."""
    comp_genre = df[['Genre'] + REGIONAL_SALES_COLUMNS]
    comp_map = comp_genre.groupby(by=['Genre']).sum().reset_index()
    comp_table = pd.melt(
        comp_map, 
        id_vars=['Genre'], 
        value_vars=REGIONAL_SALES_COLUMNS,
        var_name='Sale_Area', 
        value_name='Sale_Price'
    )
    
    plt.figure(figsize=FIGURE_SIZE_LARGE)
    sns.barplot(x='Genre', y='Sale_Price', hue='Sale_Area', data=comp_table)
    plt.title("Regional Sales Comparison by Genre")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_regional_totals(df):
    """Plot total revenue by region."""
    top_sale_reg = df[REGIONAL_SALES_COLUMNS].sum().reset_index()
    top_sale_reg = top_sale_reg.rename(columns={"index": "region", 0: "sale"})
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Bar plot
    sns.barplot(x='region', y='sale', data=top_sale_reg, ax=axes[0])
    axes[0].set_title("Total Sales by Region")
    axes[0].set_xlabel("Region")
    axes[0].set_ylabel("Sales (Millions)")
    
    # Pie chart
    axes[1].pie(
        top_sale_reg['sale'], 
        labels=top_sale_reg['region'], 
        autopct='%1.1f%%', 
        shadow=True, 
        startangle=90
    )
    axes[1].set_title("Sales Distribution by Region")
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# PUBLISHER ANALYSIS
# =============================================================================

def plot_top_publishers(df, top_n=20):
    """Plot top publishers by game count."""
    plt.figure(figsize=FIGURE_SIZE_LARGE)
    order = df.groupby(by=['Publisher'])['Year'].count().sort_values(ascending=False).iloc[:top_n].index
    sns.countplot(x="Publisher", data=df, order=order)
    plt.xticks(rotation=90)
    plt.title(f"Top {top_n} Publishers by Game Count")
    plt.xlabel("Publisher")
    plt.ylabel("Number of Games")
    plt.tight_layout()
    plt.show()


def plot_publisher_sales(df, top_n=20):
    """Plot top publishers by global sales."""
    sale_pbl = df[['Publisher', 'Global_Sales']]
    sale_pbl = sale_pbl.groupby('Publisher')['Global_Sales'].sum()
    sale_pbl = sale_pbl.sort_values(ascending=False).head(top_n)
    sale_pbl = pd.DataFrame(sale_pbl).reset_index()
    
    plt.figure(figsize=FIGURE_SIZE_LARGE)
    sns.barplot(x='Publisher', y='Global_Sales', data=sale_pbl)
    plt.xticks(rotation=90)
    plt.title(f"Top {top_n} Publishers by Global Sales")
    plt.xlabel("Publisher")
    plt.ylabel("Global Sales (Millions)")
    plt.tight_layout()
    plt.show()


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def plot_correlation_matrix(df):
    """Plot correlation heatmap for numerical columns."""
    plt.figure(figsize=(13, 10))
    numeric_df = df.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(), cmap="Blues", annot=True, linewidth=3)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


# =============================================================================
# DISTRIBUTION ANALYSIS
# =============================================================================

def plot_sales_distributions(df):
    """Plot sales distributions with gamma fit."""
    plt.figure(figsize=(25, 30))
    for i, column in enumerate(SALES_COLUMNS):
        plt.subplot(3, 2, i + 1)
        sns.histplot(df[column], bins=20, kde=False, stat='density')
        plt.title(f"{column} Distribution")
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN EDA FUNCTION
# =============================================================================

def run_full_eda(df):
    """
    Execute the complete Exploratory Data Analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset for analysis.
    """
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    print("\n[EDA] Genre Distribution Analysis...")
    plot_genre_distribution(df)
    plot_genre_sales(df)
    
    print("\n[EDA] Yearly Trends Analysis...")
    plot_yearly_releases(df)
    plot_yearly_sales(df)
    
    print("\n[EDA] Platform Analysis...")
    plot_platform_sales(df)
    
    print("\n[EDA] Regional Comparison...")
    plot_regional_comparison_by_genre(df)
    plot_regional_totals(df)
    
    print("\n[EDA] Publisher Analysis...")
    plot_top_publishers(df)
    plot_publisher_sales(df)
    
    print("\n[EDA] Correlation Analysis...")
    plot_correlation_matrix(df)
    
    print("\n[EDA] Complete!")


if __name__ == "__main__":
    # Test the module
    from data_collection import load_data
    from data_preprocessing import preprocess_pipeline
    
    df = load_data()
    df_processed = preprocess_pipeline(df)
    run_full_eda(df_processed)
