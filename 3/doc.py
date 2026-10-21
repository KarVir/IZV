#!/usr/bin/python3.10
# coding=utf-8

"""
IZV část 3 projektu- Vlastní analýza.

Autor: Karolína Virágová, xvirag00.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def make_df(df_accidents: pd.DataFrame, df_vehicles: pd.DataFrame):
    """
    Spojuje data o nehodách a vozidlech na základě sloupce 'p1',
    filtruje nehody pod vlivem alkoholu a odstraňuje neúplná data.

    Args:
        df_accidents (pd.DataFrame): DataFrame obsahující informace o nehodách.
        df_vehicles (pd.DataFrame): DataFrame obsahující informace o vozidlech.

    Returns:
        pd.DataFrame: DataFrame obsahující data.
    """
    # Spojení DataFrame na základě 'p1'
    df = pd.merge(df_accidents, df_vehicles, on="p1", how="inner")
    print("Celkový počet nehod:", len(df))
    total_accidents = len(df)

    # Filtrování nehod pod vlivem alkoholu (p11 >= 4)
    df = df[df["p11"] >= 4].dropna(subset=["p11"])
    print("Počet nehod pod vlivem alkoholu:", len(df))
    percent = (len(df) / total_accidents) * 100
    print(
        f"Procento nehod pod vlivem alkoholu: {percent:.2f}%"
    )

    df = df.dropna(subset=["p36"])
    return df


def generate_latex_table(df: pd.DataFrame) -> str:
    """
    Generuje LaTeX tabulku počtu nehod podle typu komunikace.

    Args:
        df (pd.DataFrame): Filtrovaný DataFrame obsahující údaje o nehodách.
    """
    # Mapa pro hodnoty 'p36' (druh pozemní komunikace)
    communication_types = {
        0: "Dálnice",
        1: "Silnice 1. třídy",
        2: "Silnice 2. třídy",
        3: "Silnice 3. třídy",
        4: "Uzel",
        5: "Komunikace sledovaná",
        6: "Komunikace místní",
        7: "Polní a lesní cesty",
        8: "Ostatní",
    }

    # Počet nehod podle druhu komunikace
    accidents_type = df["p36"].value_counts()
    accidents_type = accidents_type.rename(index=communication_types)

    # Generování LaTeX tabulky
    latex_output = accidents_type.to_latex(
        header=False, index=True, escape=False
    )

    latex_output = latex_output.replace(
        "p36 &", "Druh komunikace & Počet nehod"
        )

    print("\nGenerovaná LaTeX tabulka:")
    print(latex_output)
    return latex_output


def plot_alcohol(df: pd.DataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    Funkce pro zobrazení počtu nehod pod vlivem alkoholu v jednotlivých krajích
    a na různých typech komunikace.

    Args:
        df (pd.DataFrame): DataFrame obsahující informace o nehodách.
        fig_location (str): Cesta k souboru, kam bude graf uložen.
        show_figure (bool): Pokud je True, graf bude zobrazen.
    """

    # Skupinování podle regionu a typu komunikace
    accidents_grouped = (
        df.groupby(["region", "p36"]).size().reset_index(name="accident_count")
    )

    # Mapa pro typy komunikace
    communication_types = {
        0: "Dálnice",
        1: "Silnice 1. třídy",
        2: "Silnice 2. třídy",
        3: "Silnice 3. třídy",
        4: "Uzel",
        5: "Komunikace sledovaná",
        6: "Komunikace místní",
        7: "Polní a lesní cesty",
        8: "Ostatní",
    }

    # Převedení kódů na názvy komunikací
    accidents_grouped["communication_type"] = accidents_grouped["p36"].map(
        communication_types
    )

    plt.figure(figsize=(18, 10))

    ax = sns.barplot(
        data=accidents_grouped,
        x="region",
        y="accident_count",
        hue="communication_type",
        palette="muted",
        linewidth=0,
        width=0.95,
    )

    ax.set_xlabel("Kraj", fontsize=16, labelpad=10)

    ax.set_ylabel("Počet nehod", fontsize=16, labelpad=10)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.7)

    plt.legend(title="Typ komunikace", fontsize=18, title_fontsize=20)

    if fig_location:
        ax.figure.savefig(fig_location, bbox_inches="tight")
    if show_figure:
        ax.figure.show()


def plot_visible(df: pd.DataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    Funkce pro zobrazení podílu počtu nehod pod vlivem alkoholu
    a podle viditelnosti.

    Args:
        df (pd.DataFrame): DataFrame obsahující informace o nehodách.
        fig_location (str): Cesta k souboru, kam bude graf uložen.
        show_figure (bool): Pokud je True, graf bude zobrazen.
    """
    # Mapa viditelnosti
    visibility_types = {
        1: "Ve dne, viditelnost nezhoršená",
        2: "Ve dne, zhoršená viditelnost (svítání, soumrak)",
        3: "Ve dne, zhoršená viditelnost povětrnostními podmínkami",
        4: "V noci, s veřejným osvětlením, nezhoršená viditelnost",
        5: "V noci, s veřejným osvětlením, zhoršená viditelnost",
        6: "V noci, bez veřejného osvětlení, nezhoršená viditelnost",
        7: "V noci, bez veřejného osvětlení, zhoršená viditelnost",
    }

    df = df.dropna(subset=["p19"])
    df["visibility"] = df["p19"].map(visibility_types)

    # Skupinování dat podle viditelnosti a získání počtů nehod
    visibility_counts = df["visibility"].value_counts()

    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        visibility_counts,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("Set3", len(visibility_counts)),
        wedgeprops={"edgecolor": "black", "linewidth": 1.5},
        pctdistance=1.15,
        textprops={
            "fontsize": 18,
            "rotation": 45,
        },
    )

    plt.legend(
        wedges,
        visibility_counts.index,
        title="Podmínky viditelnosti",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=18,
        title_fontsize=20,
    )

    if fig_location:
        plt.savefig(fig_location, bbox_inches="tight")
    if show_figure:
        plt.show()


if __name__ == "__main__":
    df_accidents = pd.read_pickle("accidents.pkl.gz", compression="gzip")
    df_vehicles = pd.read_pickle("vehicles.pkl.gz", compression="gzip")

    df = make_df(df_accidents, df_vehicles)
    generate_latex_table(df)
    plot_alcohol(df, "fig.png")
    plot_visible(df, "fig_visible.png")
